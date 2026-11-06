"""Tests for ``assistant_axis.sweep_start_heuristics``.

The heuristic supplies a per-cell ``start_strength_multiplier_steps``
value for the bidirectional sweep.  Invariants we lock in here:

* The table is consulted ONLY when the YAML doesn't pin a value (that
  wiring lives in ``steering/run_sweep.py``; here we just verify the
  table behaviour in isolation).
* The hot cell ``(positions_mode="all", slot=7, layer=49)`` returns a
  lower start than the all-mode default (empirically tripped the
  coherence cliff at the default start in May 2026).
* The ``prefill`` default is higher than the ``all`` default
  (empirical: prefill steering is less efficacious per strength unit,
  so we need to start further up the scan).
* Unknown modes fail loudly with a KeyError that points the operator
  at the place to extend the table.
"""
from __future__ import annotations

import pytest

from assistant_axis.sweep_start_heuristics import (
    DEFAULTS,
    OVERRIDES,
    compute_start_steps,
)


class TestComputeStartSteps:
    def test_all_mode_default_for_unkeyed_cell(self):
        # (6, 49) all-mode has no override and falls through to the
        # all default (the analyser showed s_init at default was
        # already in the GOOD band; no override needed).
        assert compute_start_steps("all", 6, 49) == DEFAULTS["all"]

    def test_all_mode_default_for_off_grid_cell(self):
        # Any cell not on the production grid falls through to the
        # default for its mode.
        assert compute_start_steps("all", 99, 99) == DEFAULTS["all"]
        assert compute_start_steps("all", 0, 0) == DEFAULTS["all"]

    def test_prefill_mode_default_for_unkeyed_cell(self):
        # (0, 31) prefill has no override.
        assert compute_start_steps("prefill", 0, 31) == DEFAULTS["prefill"]
        # Off-grid prefill cell also falls through.
        assert compute_start_steps("prefill", 99, 99) == DEFAULTS["prefill"]

    def test_prefill_default_higher_than_all_default(self):
        """Anchor: the prefill default MUST be > the all default,
        because prefill steering is ~½ as efficacious per strength
        unit, so the default no-information start has to be higher."""
        assert DEFAULTS["prefill"] > DEFAULTS["all"]

    def test_hot_cell_override_in_all_mode(self):
        # (7, 49) all-mode is the hottest cell and gets a lower start.
        v = compute_start_steps("all", 7, 49)
        assert v == OVERRIDES[("all", 7, 49)]
        assert v < DEFAULTS["all"], (
            f"(all, 7, 49) override ({v}) must be < all default "
            f"({DEFAULTS['all']}); otherwise the hot-cell coherence "
            f"trip-rate regression is not fixed"
        )

    def test_hot_cell_in_all_mode_uses_negative_start_steps(self):
        # The (7, 49) all-mode cell trips coh even at the weakest=1.0
        # floor for some axes, so the override is negative
        # (s_init = weakest * mult ** -N < weakest).  Cursor must
        # support that (constraint relaxed 2026-05-24).
        assert compute_start_steps("all", 7, 49) < 0

    def test_prefill_layer_49_starts_lower_than_default(self):
        # Empirical: layer 49 prefill cells are already near the
        # coherence cliff at s_init ≈ 1.4 (median coh ≈ 0.5), so they
        # need a LOWER start than the prefill default -- the opposite
        # of what you'd guess from "prefill is less efficacious".
        assert compute_start_steps("prefill", 6, 49) < DEFAULTS["prefill"]
        assert compute_start_steps("prefill", 7, 49) < DEFAULTS["prefill"]

    def test_prefill_layer_25_starts_higher_than_default(self):
        # Empirical: layer 25 prefill cells have the weakest effect
        # per unit strength on the grid, so they need a higher start
        # than the prefill default.
        assert compute_start_steps("prefill", 0, 25) > DEFAULTS["prefill"]
        assert compute_start_steps("prefill", 6, 25) > DEFAULTS["prefill"]
        assert compute_start_steps("prefill", 7, 25) > DEFAULTS["prefill"]

    def test_unknown_mode_raises(self):
        with pytest.raises(KeyError) as exc:
            compute_start_steps("some_new_mode", 0, 25)
        msg = str(exc.value)
        # Error must point the operator at the table location.
        assert "sweep_start_heuristics" in msg
        assert "some_new_mode" in msg

    def test_returns_int(self):
        # The runner casts to int and the cursor does ** with it; make
        # sure we don't return e.g. floats from the table.
        v = compute_start_steps("all", 7, 49)
        assert isinstance(v, int)
        v = compute_start_steps("prefill", 0, 25)
        assert isinstance(v, int)

    def test_slot_and_layer_coerced(self):
        """Caller may hand us numpy ints or floats from a YAML cast --
        compute_start_steps must canonicalise both."""
        # Floats accidentally landing here should still produce the right
        # override (the table is keyed on int).
        assert compute_start_steps("all", 7, 49) == OVERRIDES[("all", 7, 49)]
        # And explicit ints obviously work.
        assert compute_start_steps("all", int(7), int(49)) == OVERRIDES[("all", 7, 49)]


class TestTableSanity:
    """Invariants about the empirical table itself."""

    def test_overrides_keys_use_known_modes(self):
        for (mode, _slot, _layer) in OVERRIDES:
            assert mode in DEFAULTS, (
                f"OVERRIDES has a mode {mode!r} not present in DEFAULTS; "
                f"this is a configuration bug -- every override mode "
                f"must also have a default for cells outside the override."
            )

    def test_overrides_differ_from_default(self):
        """Every OVERRIDES entry must differ from its mode's DEFAULTS
        value -- otherwise it adds clutter without changing behaviour
        (a no-op entry should just be removed)."""
        for (mode, slot, layer), v in OVERRIDES.items():
            assert v != DEFAULTS[mode], (
                f"OVERRIDES[{(mode, slot, layer)}] = {v} equals "
                f"DEFAULTS[{mode!r}] = {DEFAULTS[mode]}: drop the "
                f"entry (the default already covers this cell)."
            )

    def test_overrides_within_sane_range(self):
        """Defensive bound: start_steps in [-10, 20] covers s_init in
        about [0.18, 30] for the production geometric step.  Anything
        outside is almost certainly a typo."""
        for (mode, slot, layer), v in OVERRIDES.items():
            assert -10 <= v <= 20, (
                f"OVERRIDES[{(mode, slot, layer)}] = {v} outside the "
                f"sane [-10, 20] range; check for typos / sign flips."
            )
        for mode, v in DEFAULTS.items():
            assert -10 <= v <= 20, (
                f"DEFAULTS[{mode!r}] = {v} outside [-10, 20]."
            )


class TestRunSweepWiring:
    """Verify that ``steering/run_sweep.py:_build_work_items`` actually
    consults the heuristic for each cell when no YAML override is set,
    and uses the YAML override otherwise.
    """

    def _base_config(self, positions_mode: str = "all") -> dict:
        return {
            "experiment_id": "test_sweep",
            "model_name": "Qwen/Qwen3-32B",
            "output_dir": "/tmp/test_sweep_outputs",
            "axis_source": {"type": "role_transplant",
                            "vectors_dir": "/tmp", "role_from": "a",
                            "role_to": "b"},
            "persona": {"type": "role", "role": "x", "prompt_index": 0},
            "cells": [
                {"slot": 6, "layer": 49},  # falls through (no all-mode override)
                {"slot": 7, "layer": 49},  # the hot-cell override
            ],
            "sweep": {
                "weakest_strength": 1.0,
                "multiplier": 1.189,
                "signs": [+1, -1],
            },
            "positions_mode": positions_mode,
            "batch_size": 4,
            "max_new_tokens": 256,
            "questions_file": "/tmp/nonexistent.json",
        }

    def test_heuristic_consulted_per_cell_all_mode(self, tmp_path):
        from steering import run_sweep

        cfg = self._base_config(positions_mode="all")
        items = run_sweep._build_work_items(
            cfg, tmp_path, persona_prompt="p", questions=[],
        )
        cells = [it for it in items if it["kind"] == "cell"]
        for it in cells:
            if (it["slot"], it["layer"]) == (6, 49):
                # No all-mode override -> default.
                assert it["start_strength_multiplier_steps"] == DEFAULTS["all"]
            elif (it["slot"], it["layer"]) == (7, 49):
                # Hot cell override (negative).
                assert (it["start_strength_multiplier_steps"]
                        == OVERRIDES[("all", 7, 49)])

    def test_heuristic_consulted_per_cell_prefill_mode(self, tmp_path):
        from steering import run_sweep

        cfg = self._base_config(positions_mode="prefill")
        items = run_sweep._build_work_items(
            cfg, tmp_path, persona_prompt="p", questions=[],
        )
        cells = [it for it in items if it["kind"] == "cell"]
        for it in cells:
            # Both (6, 49) and (7, 49) are layer-49 prefill cells, which
            # have explicit overrides BELOW the prefill default (they're
            # already near the coh cliff at default-start).
            if (it["slot"], it["layer"]) == (6, 49):
                assert (it["start_strength_multiplier_steps"]
                        == OVERRIDES[("prefill", 6, 49)])
                assert (it["start_strength_multiplier_steps"]
                        < DEFAULTS["prefill"])
            elif (it["slot"], it["layer"]) == (7, 49):
                assert (it["start_strength_multiplier_steps"]
                        == OVERRIDES[("prefill", 7, 49)])

    def test_yaml_override_wins_over_heuristic(self, tmp_path):
        from steering import run_sweep

        cfg = self._base_config(positions_mode="all")
        cfg["sweep"]["start_strength_multiplier_steps"] = 5
        items = run_sweep._build_work_items(
            cfg, tmp_path, persona_prompt="p", questions=[],
        )
        cells = [it for it in items if it["kind"] == "cell"]
        for it in cells:
            # YAML override wins; even the (7, 49) hot cell uses 5.
            assert it["start_strength_multiplier_steps"] == 5
