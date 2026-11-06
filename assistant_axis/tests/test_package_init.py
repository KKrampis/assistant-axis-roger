"""assistant_axis/__init__.py loads its public names lazily (PEP 562).

Review P1 D5 / walk-through P8.3 (2026-10-12): the eager __init__ made
``import assistant_axis.entity_id`` cost ~1.3 s and ~2,100 modules (torch,
matplotlib.pyplot, sklearn, scipy, plotly), minutes on the pod's network
volume.  These tests pin the light import, and that the public surface
(``__all__``, every name, the submodule attributes the eager file created)
is unchanged.

The import-cost checks run in a fresh interpreter, since this test session
has long since imported torch.
"""
from __future__ import annotations

import importlib
import subprocess
import sys

import pytest

import assistant_axis

HEAVY = ("torch", "matplotlib", "sklearn", "scipy", "plotly")


def _modules_after(code: str) -> set[str]:
    out = subprocess.run(
        [sys.executable, "-c", code + "\nimport sys\nprint('\\n'.join(sys.modules))"],
        capture_output=True, text=True, check=True)
    return set(out.stdout.split())


@pytest.mark.parametrize("stmt", [
    "import assistant_axis.entity_id",
    "import assistant_axis",
    "from assistant_axis import entity_id, judge_label, response_subdir, json_metadata",
    "import assistant_axis.judge, assistant_axis.judge_pricing, assistant_axis.atomic_io, "
    "assistant_axis.openai_batch",          # what pipeline/3_judge.py loads
    "import assistant_axis.model_families",  # the family table (multi-model plan, P7)
])
def test_light_imports_load_no_heavy_library(stmt):
    loaded = _modules_after(stmt)
    heavy = sorted(m for m in loaded if m.split(".")[0] in HEAVY)
    assert not heavy, f"{stmt!r} imported {heavy[:5]}"


def test_a_heavy_name_still_loads_on_demand():
    loaded = _modules_after("from assistant_axis import load_axis")
    assert "torch" in loaded


def test_every_public_name_resolves_to_its_submodules_object():
    for name in assistant_axis.__all__:
        value = getattr(assistant_axis, name)
        module = assistant_axis._LAZY_NAMES.get(name, "entity_id")
        assert value is getattr(importlib.import_module(f"assistant_axis.{module}"), name), name


def test_all_is_unchanged():
    """The 58 names the eager file imported and exported (its __all__ is
    unchanged, verbatim; checked against HEAD when this was written)."""
    assert len(assistant_axis.__all__) == len(set(assistant_axis.__all__)) == 58
    assert set(assistant_axis.__all__) == set(assistant_axis._LAZY_NAMES) | {
        "EntityId", "KIND_R", "KIND_T", "KIND_LONG", "entity_id", "parse_entity_id",
        "is_entity_id", "display_label", "kind_short", "kind_long", "normalize_to_file_name",
        "display_form_name", "corpus_display_name", "judge_label", "judge_negative_label",
        "judge_form_of_label", "judge_form_of_negative_label"}


def test_entity_id_is_the_function_even_after_importing_the_submodule():
    """The function and its module share a name; the package attribute
    stays the function, as with the eager file."""
    importlib.import_module("assistant_axis.entity_id")
    from assistant_axis import entity_id
    assert callable(entity_id) and entity_id("patient", "roles") == "patient|R"
    assert sys.modules["assistant_axis.entity_id"].entity_id is entity_id


def test_submodules_the_eager_file_imported_are_attributes():
    for sub in ("models", "axis", "generation", "steering", "pca", "plot_metadata",
                "plot_palette", "pair_list_cohort", "judge_batch"):
        assert getattr(assistant_axis, sub) is importlib.import_module(f"assistant_axis.{sub}")
    assert "load_axis" in dir(assistant_axis) and "axis" in dir(assistant_axis)


def test_an_unknown_name_is_an_attribute_error():
    with pytest.raises(AttributeError):
        assistant_axis.no_such_name     # noqa: B018
    assert not hasattr(assistant_axis, "no_such_name")


def test_from_import_of_another_submodule_still_works():
    """``from assistant_axis import deferral_registry`` (not a lazy name)
    falls through __getattr__'s AttributeError to the normal submodule
    import."""
    from assistant_axis import deferral_registry
    assert deferral_registry.__name__ == "assistant_axis.deferral_registry"
