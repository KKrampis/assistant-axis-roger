"""scripts/judge_instruction_12way.py's Jev arms (2026-11-04): Jev 1x and 2x, the trio + Jev and Luna + Haiku + Jev at
1x and 2x, Jev's own cell names, and --without_jev leaving them out."""
import importlib.util
from pathlib import Path


def _mod():
    path = Path(__file__).resolve().parents[1] / "judge_instruction_12way.py"
    spec = importlib.util.spec_from_file_location("judge_instruction_12way", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_jev_arms_and_cells():
    mod = _mod()
    for name, mode in mod.MODES.items():
        mod.WITH_JEV = True
        a = mod.arms(mode)
        jev = {k: v for k, v in a.items() if mod.JEV in v[1]}
        labels = sorted(v[0] for v in jev.values())
        assert labels == sorted(["Jev 1x", "Jev 2x", "Luna + Haiku + Gemini + Jev 1x", "Luna + Haiku + Gemini + Jev 2x",
                                 "Luna + Haiku + Jev 1x", "Luna + Haiku + Jev 2x"]), name
        assert mod.passes_of(mode)[mod.JEV] == 2
        # the Jev sets run as arrows from 1x to 2x like the others
        assert any(set(run) == {k for k, v in jev.items() if v[0].startswith("Jev ")} for run in mod.series(mode))
        mod.WITH_JEV = False
        assert not any(mod.JEV in v[1] for v in mod.arms(mode).values()) and mod.JEV not in mod.passes_of(mode)
    mod.WITH_JEV = True
    I, Dm = mod.MODES["instructions"], mod.MODES["descriptions"]
    mod.BALANCED = False            # --directions own: the cells in the axes' own order
    assert mod.cell_of(I, mod.JEV, 7) == "jev-1.14.0@ji_noul_relative_n160_r7"
    assert mod.cell_of(Dm, mod.JEV, 1) == "jev-1.14.0@jd_noul_relative_n160s2"
    assert mod.cell_of(I, "gpt-5.6-luna", 3) == "gpt-5.6-luna@i1_p0n20r3"
    mod.BALANCED = True             # the default: the balanced cells
    assert mod.cell_of(I, mod.JEV, 7) == "jev-1.14.0@ji_noul_relative_n160d7"
    assert mod.cell_of(Dm, mod.JEV, 1) == "jev-1.14.0@jd_noul_relative_n160d2"
    assert mod.cell_of(I, "claude-haiku-5-5", 3) == "claude-haiku-5-5@i1_p0n20d3"
    assert mod.cell_of(Dm, "gpt-5.6-luna", 3) == "gpt-5.6-luna@b10n20d4"
    assert mod.cell_of(I, "gpt-5.6-luna", 12) == "gpt-5.6-luna@i1_p0n20r12"       # Luna's passes 3-4: own order


def test_balanced_choices_take_as_many_passes_each_way():
    mod = _mod()
    mod.BALANCED = True
    two = mod.replicates({"gpt-5.6-luna": 2}, {"gpt-5.6-luna": 4}, 1)
    assert sorted(r["gpt-5.6-luna"] for r in two) == [(0, 1), (0, 3), (1, 2), (2, 3)]     # one of each parity
    assert len(mod.replicates({"gpt-5.6-luna": 1}, {"gpt-5.6-luna": 4}, 1)) == 4
    assert len(mod.replicates({"gpt-5.6-luna": 3}, {"gpt-5.6-luna": 4}, 1)) == 4
    mod.BALANCED = False
    assert len(mod.replicates({"gpt-5.6-luna": 2}, {"gpt-5.6-luna": 4}, 1)) == 6
    mod.BALANCED = True


def test_jev_colour():
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from results_analysis.judge_tier_cost_plot_pooled import PROVIDER_COLOR, provider_of
    assert provider_of("jev-1.14.0") == "typesafe" and "typesafe" in PROVIDER_COLOR
    assert provider_of("gpt-5.6-luna+jev-1.14.0@x") == "blend" and provider_of("claude-haiku-5-5") == "anthropic"
