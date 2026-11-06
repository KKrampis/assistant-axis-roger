"""PC round-trip ρ pipeline.

End-to-end tooling for the experiment that asks: when we auto-describe
the n-th principal component of the post-shear entity space (using
``infer_axis_description.py``), then re-score the resulting axis text
against the entities (using ``axis_judge_correlation.py``), how well
does the round-trip recover the original projection?

The pipeline has four stages, each as a script in this package:

1. ``launch_judge_runs.py``  -- compute PC directions in post-shear
   space, save per-entity projections, then launch
   ``axis_judge_correlation.py`` for every (PC × {glossary, inline} ×
   {openai, anthropic}) cell.  Assumes ``spec.json`` already exists in
   each cell directory; the upstream
   ``infer_axis_description.py`` + ``standardize_axis_spec.py`` step
   must be run first (it is **expensive**: claude-opus-4-6 with a 10K
   thinking budget per cell).

2. ``klm_sweep.py``  -- read all cached judge scores and run the
   adaptive (L, K, M) sweep with bracket-and-bisect K refinement,
   writing a JSON cache of best round-trip ρ per (PC, style).

3. ``plot_loglin.py``  -- log-x line+marker plot of best round-trip ρ
   vs PC index.  Headline figure for the experiment.

4. ``plot_histogram.py``  -- bar-chart companion that also reports the
   best (L, K, M) per cell for diagnostic inspection.

Stages 2-4 do **no** API calls and re-run cheaply; stage 1 should be
re-run only when (a) the underlying entity vectors change, (b) the
auto-describer's specs are regenerated, or (c) gaps need filling
(use ``refill_judge_gaps.py``).
"""
