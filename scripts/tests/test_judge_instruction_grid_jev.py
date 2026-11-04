"""scripts/judge_instruction_grid.py's settled method of 2026-11-04 (Roger: "(Luna + Haiku + Gemini + Jev) x2, and just do
both directions in the x2"): four judges, two passes by default, Jev's rounds through scripts/judge_jev_entities.py
with its own method, labels and direction column, and Jev's rounds in the blend."""
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace


def _grid():
    path = Path(__file__).resolve().parents[1] / "judge_instruction_grid.py"
    spec = importlib.util.spec_from_file_location("judge_instruction_grid", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_four_judges_and_two_passes_by_default():
    g = _grid()
    assert g.INSTRUCTION_JUDGES == ("gpt-5.6-luna", "claude-haiku-5-5", "gemini-3.1-flash-lite", "jev-1.14.0")
    assert [int(x) for x in g.DEFAULT_ROUNDS.split(",")] == list(range(10))


def test_jev_rounds_run_its_own_tool_and_join_the_blend(monkeypatch, tmp_path):
    g = _grid()
    ran, blends = [], []
    monkeypatch.setattr(g.subprocess, "run", lambda cmd, **k: ran.append(cmd) or SimpleNamespace(returncode=0))
    monkeypatch.setattr(g.ev, "jobs_from_axes_file", lambda *a: [])
    monkeypatch.setattr(g.avg, "average_cells", lambda out, sources, *a, **k: blends.append((out, sources))
                        or SimpleNamespace(made=0, kept=0, refused=[]))
    axes = tmp_path / "axes.tsv"; axes.write_text("a_vs_b\ta\tb\ttraits\n")
    assert g.main(["--design", "rounds", "--axes_file", str(axes), "--instructions_dir", "D"]) == 0
    jev = [c for c in ran if str(g.JEV_TOOL) in c]
    trio = [c for c in ran if str(g.TOOL) in c]
    assert len(jev) == 10 and len(trio) == 30
    for r, cmd in enumerate(sorted(jev, key=lambda c: int(c[c.index("--instruction_round") + 1]))):
        get = lambda flag: cmd[cmd.index(flag) + 1]
        assert get("--instruction_round") == str(r) and get("--shuffle_seed") == str(r + 1)
        assert get("--cell_label") == f"@ji_noul_relative_n160d{r}" and get("--directions") == "instructions"
        assert get("--question") == "noul" and get("--entities_per_call") == "160" and get("--mode") == "instructions"
        assert get("--model") == get("--expect_model") == "jev-1.14.0"     # pinned
    out, sources = blends[0]
    assert out == "gpt-5.6-luna+claude-haiku-5-5+gemini-3.1-flash-lite+jev-1.14.0@i1_p0n20dx2"
    assert len(sources) == 40 and "jev-1.14.0@ji_noul_relative_n160d9" in sources and "gpt-5.6-luna@i1_p0n20d0" in sources


def test_jev_is_refused_in_the_sets_design(tmp_path):
    g = _grid()
    import pytest
    with pytest.raises(SystemExit):
        g.main(["--design", "sets", "--axes_file", "x", "--instructions_dir", "D", "--judges", "jev-1.14.0"])
