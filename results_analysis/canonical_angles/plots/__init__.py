"""Plot wrappers for the canonical-angles tool.

Each module here is a small CLI script that:

1. Parses CLI flags
2. Builds a list of :class:`~results_analysis.canonical_angles.core.CASpec`
3. Calls :func:`~results_analysis.canonical_angles.core.compute_ca_grid`
4. Composes :mod:`~results_analysis.canonical_angles.plot_helpers` to format
   the result for its specific plot family
"""
