"""Tests for steering/run_sweep.py's pure helpers.

The worker-process body (model loading, run_steering_cell call) needs a
GPU and is exercised end-to-end on RunPod, not in unit tests.  Here we
cover the config parsing, persona prompt assembly, and work-item
construction.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_dispatcher_module():
    """Load steering/run_sweep.py via importlib.

    Done explicitly (rather than `from steering.run_sweep import ...`)
    so the test file doesn't need a top-level `steering` package init,
    and so the module name in tracebacks is unambiguous.
    """
    spec = importlib.util.spec_from_file_location(
        "_steering_run_sweep",
        REPO_ROOT / "steering" / "run_sweep.py",
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def mod():
    return _load_dispatcher_module()


# ---------------------------------------------------------------------------
# _load_questions
# ---------------------------------------------------------------------------

class TestLoadQuestions:
    def test_flat_list(self, mod, tmp_path):
        p = tmp_path / "q.json"
        p.write_text(json.dumps(["What is X?", "Why Y?"]))
        assert mod._load_questions(p) == ["What is X?", "Why Y?"]

    def test_dict_with_key(self, mod, tmp_path):
        p = tmp_path / "q.json"
        p.write_text(json.dumps({"questions": ["q1", "q2"], "meta": "ignored"}))
        assert mod._load_questions(p) == ["q1", "q2"]

    def test_invalid_shape_raises(self, mod, tmp_path):
        p = tmp_path / "q.json"
        p.write_text(json.dumps({"foo": "bar"}))
        with pytest.raises(ValueError):
            mod._load_questions(p)

    def test_non_string_entries_raise(self, mod, tmp_path):
        p = tmp_path / "q.json"
        p.write_text(json.dumps(["q1", 42]))
        with pytest.raises(ValueError):
            mod._load_questions(p)


# ---------------------------------------------------------------------------
# _build_persona_system_prompt
# ---------------------------------------------------------------------------

class TestPersonaPrompt:
    def test_role_uses_first_instruction_pos(self, mod):
        # Real role file in the repo:
        out = mod._build_persona_system_prompt(
            {"type": "role", "role": "historian"},
            instructions_dir=REPO_ROOT / "data",
        )
        assert "historian" in out.lower()
        # Should match instruction[0].pos exactly
        with open(REPO_ROOT / "data" / "roles" / "instructions" / "historian.json") as f:
            data = json.load(f)
        assert out == data["instruction"][0]["pos"]

    def test_role_with_explicit_prompt_index(self, mod):
        out = mod._build_persona_system_prompt(
            {"type": "role", "role": "historian", "prompt_index": 2},
            instructions_dir=REPO_ROOT / "data",
        )
        with open(REPO_ROOT / "data" / "roles" / "instructions" / "historian.json") as f:
            data = json.load(f)
        assert out == data["instruction"][2]["pos"]

    def test_combination_concatenates_with_newline(self, mod):
        out = mod._build_persona_system_prompt(
            {
                "type": "combination",
                "role": "historian",
                "traits": ["stoic"],
            },
            instructions_dir=REPO_ROOT / "data",
        )
        with open(REPO_ROOT / "data" / "roles" / "instructions" / "historian.json") as f:
            role_inst = json.load(f)["instruction"][0]["pos"]
        with open(REPO_ROOT / "data" / "traits" / "instructions" / "stoic.json") as f:
            trait_inst = json.load(f)["instruction"][0]["pos"]
        assert out == f"{role_inst}\n{trait_inst}"

    def test_combination_requires_traits(self, mod):
        with pytest.raises(ValueError, match="trait"):
            mod._build_persona_system_prompt(
                {"type": "combination", "role": "historian", "traits": []},
                instructions_dir=REPO_ROOT / "data",
            )

    def test_unknown_type_raises(self, mod):
        with pytest.raises(ValueError, match="persona.type"):
            mod._build_persona_system_prompt(
                {"type": "alien"}, instructions_dir=REPO_ROOT / "data",
            )

    def test_missing_role_file_raises(self, mod, tmp_path):
        # Fake instructions dir with nothing in it
        (tmp_path / "roles" / "instructions").mkdir(parents=True)
        with pytest.raises(FileNotFoundError):
            mod._build_persona_system_prompt(
                {"type": "role", "role": "nonexistent"},
                instructions_dir=tmp_path,
            )


# ---------------------------------------------------------------------------
# _build_work_items
# ---------------------------------------------------------------------------

class TestBuildWorkItems:
    def _basic_config(self):
        return {
            "experiment_id": "test_exp",
            "model_name": "Qwen/Qwen3-32B",
            "axis_source": {
                "type": "role_transplant",
                "vectors_dir": "/tmp/vecs",
                "role_from": "angel",
                "role_to": "demon",
            },
            "cells": [
                {"slot": 0, "layer": 26},
                {"slot": 3, "layer": 25},
            ],
            "sweep": {
                "weakest_strength": 1.0,
                "max_strength": 4.0,
                "multiplier": 2.0,
                "signs": [+1, -1],
            },
            "batch_size": 4,
            "max_new_tokens": 128,
            "positions_mode": "all",
        }

    def test_baseline_item_first(self, mod, tmp_path):
        items = mod._build_work_items(
            self._basic_config(), tmp_path, "you are a historian", ["q1", "q2"]
        )
        assert items[0]["kind"] == "baselines"
        assert items[0]["persona"] == "you are a historian"
        assert items[0]["questions"] == ["q1", "q2"]

    def test_one_cell_per_slot_layer_sign(self, mod, tmp_path):
        items = mod._build_work_items(
            self._basic_config(), tmp_path, "p", ["q"]
        )
        cells = [it for it in items if it["kind"] == "cell"]
        assert len(cells) == 4  # 2 cells x 2 signs
        signs = [(c["slot"], c["layer"], c["sign"]) for c in cells]
        assert sorted(signs) == [
            (0, 26, -1), (0, 26, +1), (3, 25, -1), (3, 25, +1),
        ]

    def test_cell_dir_name_format(self, mod, tmp_path):
        items = mod._build_work_items(
            self._basic_config(), tmp_path, "p", ["q"]
        )
        cells = [it for it in items if it["kind"] == "cell"]
        cell_dirs = {Path(c["cell_dir"]).name for c in cells}
        assert cell_dirs == {
            "s0_l26_+1", "s0_l26_-1", "s3_l25_+1", "s3_l25_-1",
        }

    def test_per_cell_weakest_override(self, mod, tmp_path):
        cfg = self._basic_config()
        cfg["cells"] = [
            {"slot": 0, "layer": 26},                              # default 1.0
            {"slot": 3, "layer": 25, "weakest_strength": 0.5},     # override
        ]
        items = mod._build_work_items(cfg, tmp_path, "p", ["q"])
        cells = [it for it in items if it["kind"] == "cell"]
        weakest_by_slot = {(c["slot"], c["layer"]): c["weakest_strength"]
                           for c in cells}
        assert weakest_by_slot[(0, 26)] == 1.0
        assert weakest_by_slot[(3, 25)] == 0.5

    def test_signs_only_positive(self, mod, tmp_path):
        cfg = self._basic_config()
        cfg["sweep"]["signs"] = [+1]
        items = mod._build_work_items(cfg, tmp_path, "p", ["q"])
        cells = [it for it in items if it["kind"] == "cell"]
        assert len(cells) == 2  # 2 cells x 1 sign
        assert all(c["sign"] == +1 for c in cells)

    def test_invalid_sign_raises(self, mod, tmp_path):
        cfg = self._basic_config()
        cfg["sweep"]["signs"] = [+1, 0]
        with pytest.raises(ValueError, match="signs"):
            mod._build_work_items(cfg, tmp_path, "p", ["q"])

    def test_empty_cells_raises(self, mod, tmp_path):
        cfg = self._basic_config()
        cfg["cells"] = []
        with pytest.raises(ValueError, match="cells"):
            mod._build_work_items(cfg, tmp_path, "p", ["q"])


class TestMultiGpuQueueOrdering:
    """The multi-config queue runner sorts all baselines to the head of
    the shared queue so N_gpus workers parallelize baselines first
    (rather than one worker stuck on a baseline while N-1 workers grab
    cells that depend on it).  This test verifies the sort key alone
    -- end-to-end multi-process behaviour is exercised on the GPU pod.
    """

    def test_baselines_sort_to_head(self, mod):
        # Mixed queue across two notional experiments, plus a stable
        # secondary order so we can assert within-config preservation.
        items = [
            {"kind": "baselines", "tag": "A"},
            {"kind": "cell", "tag": "A0"},
            {"kind": "cell", "tag": "A1"},
            {"kind": "baselines", "tag": "B"},
            {"kind": "cell", "tag": "B0"},
            {"kind": "cell", "tag": "B1"},
        ]
        # Mirror the sort key used in run_sweep.main().
        items.sort(key=lambda it: 0 if it["kind"] == "baselines" else 1)

        kinds = [it["kind"] for it in items]
        assert kinds == ["baselines", "baselines",
                         "cell", "cell", "cell", "cell"]
        # Stable: within-kind original order is preserved.
        tags = [it["tag"] for it in items]
        assert tags == ["A", "B", "A0", "A1", "B0", "B1"]


class TestSweepLogFileHandler:
    """Attach/detach pair for the per-experiment sweep.log FileHandler.
    Used by the multi-config queue runner to swap handlers as the worker
    transitions between experiments in the queue.
    """

    def test_attach_idempotent(self, mod, tmp_path):
        h1 = mod._attach_sweep_log_filehandler(tmp_path)
        h2 = mod._attach_sweep_log_filehandler(tmp_path)
        assert h1 is h2, "second attach should return the same handler"
        # cleanup
        mod._detach_sweep_log_filehandler(tmp_path)

    def test_attach_writes_to_target(self, mod, tmp_path):
        import logging
        # Root logger needs INFO level for our handler (level=INFO) to fire.
        # Save/restore so other tests aren't affected.
        prev_root_level = logging.getLogger().level
        logging.getLogger().setLevel(logging.INFO)
        try:
            mod._attach_sweep_log_filehandler(tmp_path)
            logging.getLogger("test_run_sweep").info("hello sweep.log")
            # Flush so the file is observable.
            for h in logging.getLogger().handlers:
                h.flush()
            sweep_log = tmp_path / "sweep.log"
            assert sweep_log.exists()
            assert "hello sweep.log" in sweep_log.read_text()
        finally:
            mod._detach_sweep_log_filehandler(tmp_path)
            logging.getLogger().setLevel(prev_root_level)

    def test_detach_removes_handler(self, mod, tmp_path):
        import logging
        h = mod._attach_sweep_log_filehandler(tmp_path)
        root_handlers = logging.getLogger().handlers
        assert h in root_handlers
        mod._detach_sweep_log_filehandler(tmp_path)
        assert h not in logging.getLogger().handlers

    def test_detach_no_op_when_not_attached(self, mod, tmp_path):
        # Detaching a path we never attached should not raise.
        mod._detach_sweep_log_filehandler(tmp_path / "nonexistent")

    def test_swap_between_two_dirs(self, mod, tmp_path):
        d1 = tmp_path / "exp_a"
        d2 = tmp_path / "exp_b"
        h1 = mod._attach_sweep_log_filehandler(d1)
        # Now "transition" to a new experiment: detach the first, attach
        # the second.  Workers do this in their queue loop.
        mod._detach_sweep_log_filehandler(d1)
        h2 = mod._attach_sweep_log_filehandler(d2)
        assert h1 is not h2
        import logging
        root_handlers = logging.getLogger().handlers
        assert h2 in root_handlers
        assert h1 not in root_handlers
        # cleanup
        mod._detach_sweep_log_filehandler(d2)


# ---------------------------------------------------------------------------
# _resolve_hf_home — auto-detection of HF cache location
# ---------------------------------------------------------------------------

class TestResolveHfHome:
    """Verify the HF_HOME fallback search hits the right candidate first.

    The dispatcher ships with a hard-coded fallback list pointing at
    canonical RunPod NFS cache paths.  That list is sticky across
    test cases (a module-level constant), so each test uses
    ``monkeypatch`` to replace ``DEFAULT_HF_HOME_FALLBACKS`` with a
    list of tmp_path-derived candidates.  Avoids tests poking at real
    /workspace/.cache/huggingface and accidentally passing on a dev
    box that happens to have one.
    """

    def _setup(self, tmp_path, present_dirs, fallbacks, monkeypatch, mod,
               env_hf_home=None):
        for d in present_dirs:
            (tmp_path / d / "hub" / "models--foo--bar").mkdir(parents=True)
        monkeypatch.setattr(
            mod, "DEFAULT_HF_HOME_FALLBACKS",
            tuple(str(tmp_path / f) for f in fallbacks),
        )
        if env_hf_home is None:
            monkeypatch.delenv("HF_HOME", raising=False)
        else:
            monkeypatch.setenv("HF_HOME", str(tmp_path / env_hf_home))

    def test_returns_env_hf_home_when_populated(
            self, mod, tmp_path, monkeypatch):
        self._setup(tmp_path, ["env_cache"],
                    fallbacks=["wsp_cache"], monkeypatch=monkeypatch, mod=mod,
                    env_hf_home="env_cache")
        assert mod._resolve_hf_home("foo/bar") == str(tmp_path / "env_cache")

    def test_falls_back_when_env_unset(
            self, mod, tmp_path, monkeypatch):
        self._setup(tmp_path, ["wsp_cache"],
                    fallbacks=["wsp_cache"], monkeypatch=monkeypatch, mod=mod)
        assert mod._resolve_hf_home("foo/bar") == str(tmp_path / "wsp_cache")

    def test_falls_back_when_env_empty(
            self, mod, tmp_path, monkeypatch):
        # User has HF_HOME set but it's a stale path that doesn't have
        # the model.  Should skip past it to the populated fallback.
        self._setup(tmp_path, ["wsp_cache"],
                    fallbacks=["wsp_cache"], monkeypatch=monkeypatch, mod=mod,
                    env_hf_home="empty_env")
        assert mod._resolve_hf_home("foo/bar") == str(tmp_path / "wsp_cache")

    def test_first_populated_fallback_wins(
            self, mod, tmp_path, monkeypatch):
        # Both fallbacks have the model -- the FIRST one in the list
        # should win, matching priority order semantics.
        self._setup(tmp_path, ["primary", "secondary"],
                    fallbacks=["primary", "secondary"],
                    monkeypatch=monkeypatch, mod=mod)
        assert mod._resolve_hf_home("foo/bar") == str(tmp_path / "primary")

    def test_returns_none_when_nothing_found(
            self, mod, tmp_path, monkeypatch):
        # Neither env nor any fallback has the model: caller should
        # know to fail loudly rather than letting HF download.
        self._setup(tmp_path, [],
                    fallbacks=["empty1", "empty2"],
                    monkeypatch=monkeypatch, mod=mod)
        assert mod._resolve_hf_home("foo/bar") is None

    def test_model_name_with_slash(self, mod, tmp_path, monkeypatch):
        # HF stores "Org/Model" as "models--Org--Model".  The resolver
        # has to do that mapping; this is the regression guard.
        (tmp_path / "cache" / "hub" / "models--Qwen--Qwen3-32B").mkdir(
            parents=True)
        monkeypatch.setattr(
            mod, "DEFAULT_HF_HOME_FALLBACKS",
            (str(tmp_path / "cache"),),
        )
        monkeypatch.delenv("HF_HOME", raising=False)
        assert mod._resolve_hf_home("Qwen/Qwen3-32B") == str(
            tmp_path / "cache")
