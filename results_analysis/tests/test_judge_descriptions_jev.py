"""judge_descriptions.py with Jev as the fourth judge (Roger, 2026-11-04: "the new standard is (Luna + Haiku + Gemini +
Jev) x2, with the reversal in the x2"): Jev is in the default judges, its passes go to scripts/judge_jev_entities.py
with its own method and labels and the same seeds and direction passes as the others, and its passes are averaged
into its own cell and into the blend."""
from pathlib import Path
from types import SimpleNamespace

import pytest

from results_analysis import judge_descriptions as jd


def _args(**kw):
    base = dict(axes_file="axes.tsv", pair=None, pair_type=None, experiments_dir="E", instructions_dir="D",
                judge_model=None, entities_per_call=20, passes=2, first_seed=1, directions="balanced",
                name_format=jd.DEFAULT_NAME_FORMAT, transport="auto", budget_usd=None, limit=None,
                requests_per_second=None, max_in_flight=None, dry_run=False)
    base.update(kw)
    return SimpleNamespace(**base)


def test_jev_is_a_default_judge():
    assert jd.JEV_MODEL in jd.DESCRIPTION_JUDGES and jd.DESCRIPTION_JUDGES[:3] == (
        "gpt-5.6-luna", "claude-haiku-5-5", "gemini-3.1-flash-lite")


def test_jev_passes_go_to_its_own_tool_with_the_same_seeds_and_directions(monkeypatch):
    calls = []
    monkeypatch.setattr(jd, "run_tool", lambda argv, tool=jd.TOOL: calls.append((tool, list(argv))) or 0)
    assert jd.run_judge(_args(), jd.JEV_MODEL) == 0
    assert [t for t, _ in calls] == [jd.JEV_TOOL, jd.JEV_TOOL]
    for p, (_, argv) in enumerate(calls):
        get = lambda flag: argv[argv.index(flag) + 1]
        assert get("--shuffle_seed") == str(p + 1) and get("--direction_pass") == str(p)
        assert get("--cell_label") == f"@jd_noul_relative_n160d{p + 1}" and get("--entities_per_call") == "160"
        assert get("--question") == "noul" and get("--wording") == "relative"
        # requested by its versioned ID, pinned, and the answers required to come from it
        assert get("--model") == get("--expect_model") == jd.JEV_MODEL == "jev-1.14.0"
        assert get("--directions") == "descriptions" and get("--axes_file") == "axes.tsv"
    calls.clear()
    assert jd.run_judge(_args(), "gpt-5.6-luna") == 0
    assert all(t == jd.TOOL for t, _ in calls) and calls[0][1][calls[0][1].index("--cell_label") + 1] == "@b10n20d1"
    calls.clear()
    jd.run_judge(_args(directions="own", pair=("alpha", "beta"), pair_type="traits", axes_file=None, limit=5), jd.JEV_MODEL)
    argv = calls[0][1]
    assert "--directions" not in argv and argv[argv.index("--cell_label") + 1] == "@jd_noul_relative_n160s1"
    assert argv[argv.index("--pair") + 1: argv.index("--pair") + 3] == ["alpha", "beta"] and "--limit" in argv


def test_jev_passes_are_averaged_into_its_cell_and_the_blend(monkeypatch):
    made = []
    monkeypatch.setattr(jd, "jobs_of", lambda args: [])
    monkeypatch.setattr(jd.avg, "average_cells", lambda out, sources, *a, **k: made.append((out, list(sources)))
                        or SimpleNamespace(made=0, kept=0, refused=[]))
    assert jd.average_all(_args(), list(jd.DESCRIPTION_JUDGES)) == []
    outs = dict(made)
    assert outs[f"{jd.JEV_MODEL}@jd_noul_relative_n160dx2"] == [f"{jd.JEV_MODEL}@jd_noul_relative_n160d1",
                                                                f"{jd.JEV_MODEL}@jd_noul_relative_n160d2"]
    blend = next(v for k, v in outs.items() if "+" in k)
    assert f"{jd.JEV_MODEL}@jd_noul_relative_n160d2" in blend and "gpt-5.6-luna@b10n20d1" in blend and len(blend) == 8


def test_a_name_format_other_than_jevs_is_refused_with_jev(monkeypatch):
    monkeypatch.setattr(jd, "run_judge", lambda *a: 0)
    with pytest.raises(SystemExit):
        jd.main(["--axes_file", "x.tsv", "--name_format", "mechanical"])
