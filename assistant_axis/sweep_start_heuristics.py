"""Per-cell start-strength heuristic for bidirectional steering sweeps.

The bidirectional-scan starts at ``s_init = weakest * multiplier ** N``
and walks both UP (until coherence-stop) and DOWN (until effect-stop).
If ``s_init`` lands above the coherence cliff the UP walk stops
immediately and we waste cells.  If it lands below the effect threshold
the DOWN walk stops immediately and we waste cells the other way.

A single flat ``N`` (the historical default ``start_strength_multiplier_steps=2``,
giving ``s_init ≈ 1.41`` at ``weakest=1.0, mult=1.189``) is empirically
sub-optimal across the production grid: it lands TOO_LOW in ~89% of
prefill cells, TOO_LOW in ~46% of all-mode cells, and TOO_HIGH in
~25% of all-mode ``(slot=7, layer=49)`` cells.

This module supplies a small empirically-fit lookup table keyed on
``(positions_mode, slot, layer)`` plus per-mode defaults, returning
the integer ``N`` to use as ``start_strength_multiplier_steps`` for
that cell.  Higher ``N`` ⇒ higher ``s_init`` ⇒ more aggressive
starting point.  ``N`` can be negative; the cursor's actual safety
floor for ``s_init`` is ``min_strength`` (default 0.125), not zero.

Empirical derivation
====================
The numbers below come from
[`tools/analyse_start_strength.py`](../tools/analyse_start_strength.py),
which scans all production sweep records and reports, per
``(positions_mode, slot, layer, sign)``, the median ``|effect|`` and
median ``coh`` at the strength closest to the cell's configured
``s_init``.  For each ``(positions_mode, slot, layer)`` bucket we then
compute the strength range ``[s_eff_lo, s_coh_hi]`` such that the
worst-sign effect lands above ``TARGET_EFF = 0.5`` (well above the
``1/3`` eff_stop threshold) and the worst-sign coherence lands below
``TARGET_COH = 0.5`` (well below the ``1.5`` coh_stop threshold),
assuming approximately linear scaling near zero strength.  The
recommended ``s_init`` is the geometric mean of those bounds (or the
constraining endpoint when they conflict), rounded to the nearest
integer ``N`` such that ``s_init ≈ weakest * mult ** N``.

Re-run the analyser after each meaningful new sweep batch (e.g. when
the next-role campaign lands and we have ~2× more cells per bucket)
and tune the table if distributions shift.  See the "Per-cell start
strength heuristic" section of AGENT_NOTES.md for the full
justification + re-tuning recipe.

Override semantics
==================
The lookup is consulted only when the per-sweep YAML does NOT set
``sweep.start_strength_multiplier_steps`` explicitly.  Any YAML value
takes priority (so a one-off experiment can hand-tune its start).

API
===
- :func:`compute_start_steps(positions_mode, slot, layer)` — the
  single entry point.  Returns the int ``N`` to use.  Falls through
  the table → per-mode default → fail-loud-on-unknown-mode in that
  order.
"""
from __future__ import annotations

from typing import Dict, Tuple

__all__ = ["compute_start_steps", "OVERRIDES", "DEFAULTS"]


# ---------------------------------------------------------------------------
# Empirical lookup table (May 24, 2026).
# ---------------------------------------------------------------------------
#
# Tuned from the 12-axis × 14-cell × 2-mode multi-cell sweep (310 cells
# with >=1 judged record at the historical s_init = 1.413).  Entries
# below encode (slot, layer)-specific deviations from the per-mode
# default; defaults absorb the bulk of cells.
#
# Pattern observed:
#  * deeper layers (49) have stronger steering per strength unit, so
#    s_init should be LOWER (lower start_steps);
#  * shallower layers (25) have weaker effect, so s_init should be
#    HIGHER (higher start_steps);
#  * slot 0 is sometimes high-effect (because the steering hits the
#    initial sentinel tokens), sometimes ordinary -- per-(slot, layer)
#    measurement;
#  * prefill mode is generally less-efficacious per unit strength so
#    needs higher s_init, EXCEPT at layer 49 where prefill is already
#    near the coherence cliff at s≈1.4 (median coh ~0.5) and a higher
#    start would push it over.

OVERRIDES: Dict[Tuple[str, int, int], int] = {
    # ----- all-mode -----
    # (0, 25): eff slightly weak (0.29 / 0.54 across signs).  Bump 5.
    ("all", 0, 25): 5,
    # (0, 31): eff borderline (0.36 / 0.39).  Bump 4.
    ("all", 0, 31): 4,
    # (0, 49): high-effect (0.68 / 1.11) -- already in the good band,
    # but lowering to 0 reduces the marginal TOO_HIGH risk on +1.
    ("all", 0, 49): 0,
    # (6, 25): the weakest all-mode cell (eff ~0.26).  Bump 6.
    ("all", 6, 25): 6,
    # (7, 25): also weak (eff ~0.21 / 0.30).  Bump 7.
    ("all", 7, 25): 7,
    # (7, 49): the hot cell.  Median coh@1.41 is 0.43 / 1.29; the +1
    # sign trips the coh cliff in ~25% of axes at the historical start.
    # Drop to start_steps=-3 (s_init ≈ 0.59) so the median start lands
    # at coh ~0.18 / 0.54 and eff ~0.22 / 0.39 -- still above the
    # 1/3 eff threshold for the stronger sign, and the cursor walks up
    # from there.  REQUIRES start_steps_up < 0 (relaxed 2026-05-24;
    # safety floor is now min_strength).
    ("all", 7, 49): -3,

    # ----- prefill-mode -----
    # All prefill cells have weaker effect per unit strength (~50% of
    # all-mode), so most need a HIGHER start than the per-mode default.
    # EXCEPT layer 49 prefill cells, which are already near the cliff
    # at s_init=1.41 (median coh ~0.5).
    ("prefill", 0, 25): 9,   # eff 0.16 / 0.23 -> need ~5x
    ("prefill", 0, 49): 5,   # eff 0.20 / 0.25, coh 0.14 / 0.29 -> 2-3x
    ("prefill", 6, 25): 9,   # eff 0.14 / 0.16, weak; bump significantly
    ("prefill", 6, 49): 1,   # coh 0.29 / 0.57 already near cliff; lower
    ("prefill", 7, 25): 9,   # eff 0.16 / 0.21 weak
    ("prefill", 7, 49): 1,   # coh 0.57 / 0.57 near cliff; lower
    # (prefill, 0, 31) matches the prefill default below.
}

DEFAULTS: Dict[str, int] = {
    # all-mode: 3 (s_init ≈ 1.68) is one step above the historical 2
    # (s_init ≈ 1.41) and lifts most "ordinary" (no override) cells
    # squarely into GOOD.  Slot/layer combos that deviate are
    # captured in OVERRIDES above.  Falls back here for any
    # (slot, layer) the grid expands to in future.
    "all": 3,
    # prefill-mode: 6 (s_init ≈ 2.67) is the balanced default across
    # the prefill grid -- pushes the bulk of cells above the eff
    # threshold without pushing layer-49 over the coh cliff.  Cells
    # with extreme behaviour on either end get their own OVERRIDES
    # entry; this is the "no information about (slot, layer)" choice.
    "prefill": 6,
}


def compute_start_steps(positions_mode: str, slot: int, layer: int) -> int:
    """Return the recommended ``start_strength_multiplier_steps`` for a cell.

    Args:
        positions_mode: ``"all"`` or ``"prefill"`` (the steering applies
            at every token position vs only at the prefill / system
            positions).
        slot: Persona-prompt slot index (0/3/6/7 in the production grid).
        layer: Transformer layer index (25/31/49 in the production grid).

    Returns:
        Integer ``N`` such that ``s_init = weakest * multiplier ** N``
        is the recommended bidirectional-scan starting point for this
        cell, derived from the empirical
        :data:`OVERRIDES` / :data:`DEFAULTS` table.

    Raises:
        KeyError: when ``positions_mode`` is not one of the keys in
            :data:`DEFAULTS`.  Loud-fail keeps misconfigured runs out
            of production (silently treating an unknown mode as
            ``DEFAULTS["all"]`` would be a silent regression on prefill
            sweeps, which we know need a different start).
    """
    key = (positions_mode, int(slot), int(layer))
    if key in OVERRIDES:
        return OVERRIDES[key]
    try:
        return DEFAULTS[positions_mode]
    except KeyError as exc:
        valid = ", ".join(sorted(DEFAULTS))
        raise KeyError(
            f"unknown positions_mode {positions_mode!r}; expected one of "
            f"[{valid}].  Add an entry to "
            f"assistant_axis/sweep_start_heuristics.DEFAULTS if you've "
            f"introduced a new mode."
        ) from exc
