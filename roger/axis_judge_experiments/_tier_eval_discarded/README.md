# Discarded judge-tier cells

Cells moved out of the `<axis>_tier_eval/` directories so that the plotters
and the robustness tool do not pick them up.  Kept as a record of the spend.

- `*/claude-haiku-5-5@b10s1`, `*/claude-haiku-5-5@b10s2` (2026-11-01): the
  first batched percentile runs of Haiku 5.5 on the 30 development axes,
  ten entities per call at a 600-token reply cap.  Haiku writes
  about 60 tokens per entity, so 40% of the ten-entity replies were cut at
  the cap and their entities scored as missing; the runs were repeated at a
  1,500-token cap as `claude-haiku-5-5@b10s3` and `@b10s4`.  Logs:
  `../_tier_eval_logs/dev30_haiku55_b10s{1,2}_20261101T184659Z.log`.
