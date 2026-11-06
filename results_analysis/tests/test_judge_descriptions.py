"""The production entry for description judging (judge_descriptions.py)."""
import asyncio
import json

import pytest

from results_analysis import judge_descriptions as jd
from results_analysis import judge_tier_cost_eval as ev
from assistant_axis.judge_score_combine import combine_tier_judges, tier_judge_weights
from results_analysis.tests.test_judge_tier_cost_eval import (  # noqa: F401 -- the fixture
    AXES, DIRECTION_AWARE, ENTITIES, SCORES, FakeAnthropic, FakeGeminiBatch, FakeJevBatch, FakeOpenAI, FlakyOpenAI,
    _script, batch_score, world,
)


def test_the_decision_is_the_default():
    args = jd.build_parser().parse_args(["--axes_file", "x"])
    assert (args.entities_per_call, args.passes, args.judge_model) == (20, 2, None)
    # Gemini joined the default on 2026-11-03 (Roger: "Luna + Haiku + Gemini at 2x each"), Jev on 2026-11-04
    # (Roger: "the new standard is (Luna + Haiku + Gemini + Jev) x2, with the reversal in the x2")
    assert jd.DESCRIPTION_JUDGES == ("gpt-5.6-luna", "claude-haiku-5-5", "gemini-3.1-flash-lite", "jev-1.14.0")
    assert jd.DESCRIPTION_RUBRIC == "b10" and args.name_format == "judge_label" and args.transport == "auto"
    assert args.directions == "balanced"        # Roger, 2026-11-04: each judge's two passes, one each way


def test_a_pass_runs_the_tool_as_the_grid_did():
    args = jd.build_parser().parse_args(["--axes_file", "a.tsv", "--budget_usd", "5"])
    argv = jd.pass_argv(args, "gpt-5.6-luna", 2)
    flags = dict(zip(argv[::2], argv[1::2]))
    assert flags["--judge_model"] == "gpt-5.6-luna" and flags["--rubric"] == "b10" and flags["--entities_per_call"] == "20"
    assert flags["--shuffle_seed"] == "2" and flags["--cell_label"] == "@b10n20d2" and flags["--budget_usd"] == "5.0"
    assert flags["--directions"] == "descriptions" and flags["--direction_pass"] == "1"     # the second pass
    assert flags["--max_tokens"] == "1200" and flags["--axes_file"] == "a.tsv"       # the grid's cap at N=20
    assert jd.max_tokens_for("claude-haiku-5-5", 20) == 3000                          # its Haiku cap
    assert jd.max_tokens_for("gemini-3.1-flash-lite", 20) == 1200
    assert "--request_setting" not in flags                                           # no reasoning: judge.py's table


def test_own_directions_keep_the_old_labels_and_send_no_direction():
    args = jd.build_parser().parse_args(["--axes_file", "a.tsv", "--directions", "own"])
    flags = dict(zip(jd.pass_argv(args, "gpt-5.6-luna", 1)[::2], jd.pass_argv(args, "gpt-5.6-luna", 1)[1::2]))
    assert flags["--cell_label"] == "@b10n20s1" and "--directions" not in flags
    assert jd.averaged_label(20, 2, "own") == "@b10n20x2" and jd.averaged_label(20, 2, "balanced") == "@b10n20dx2"


def _in_process(monkeypatch):
    """The tools in this process, one pass at a time (they keep a run's settings in module state): the evaluator,
    and Jev's (scripts/judge_jev_entities.py) with the fake Jev."""
    monkeypatch.setattr(jd, "MAX_CONCURRENT_JUDGES", 1)
    jev = _script("judge_jev_entities")
    fake = FakeJevBatch()
    monkeypatch.setattr(jev.jrj, "make_client", lambda: fake)

    def run(argv, tool=jd.TOOL):
        if tool == jd.JEV_TOOL:
            return asyncio.run(jev.main_async(list(argv)))
        return asyncio.run(ev.main_async([*argv, "--batch_poll_seconds", "0"]))
    monkeypatch.setattr(jd, "run_tool", run)
    return fake


def _args(world, *extra):
    return ["--axes_file", str(world.axes2), "--experiments_dir", str(world.exp), "--instructions_dir", str(world.data),
            "--entities_per_call", "3", "--requests_per_second", "1000", *extra]


def test_a_run_writes_the_passes_the_averages_and_the_blend(world, monkeypatch):
    _in_process(monkeypatch)
    flex, anthropic, gemini = FakeOpenAI(), FakeAnthropic(), FakeGeminiBatch()
    monkeypatch.setattr(ev, "make_openai_client", lambda service_tier=None: flex)
    monkeypatch.setattr(ev, "make_anthropic_client", lambda: anthropic)
    monkeypatch.setattr(ev, "make_gemini_client", lambda: gemini)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-used")
    monkeypatch.setitem(DIRECTION_AWARE, "on", True)    # judges that follow the scale shown: both directions agree
    assert jd.main(_args(world)) == 0
    tier = world.exp / f"{AXES[0][0]}_tier_eval"
    names = {p.name for p in tier.iterdir()}
    # the three judges since 2026-11-03, with Jev since 2026-11-04
    four = "gpt-5.6-luna+claude-haiku-5-5+gemini-3.1-flash-lite+jev-1.14.0@b10n3dx2"
    assert {"gpt-5.6-luna@b10n3d1", "gpt-5.6-luna@b10n3d2", "gpt-5.6-luna@b10n3dx2", "claude-haiku-5-5@b10n3d1", "claude-haiku-5-5@b10n3d2",
            "claude-haiku-5-5@b10n3dx2", "gemini-3.1-flash-lite@b10n3dx2", "jev-1.14.0@jd_noul_relative_n160d1",
            "jev-1.14.0@jd_noul_relative_n160d2", "jev-1.14.0@jd_noul_relative_n160dx2", four} <= names
    for m in ("gpt-5.6-luna", "claude-haiku-5-5", "gemini-3.1-flash-lite"):    # one pass each way, by every judge
        ways = [json.loads((tier / f"{m}@b10n3d{k}" / "scores_descriptions.json").read_text())["_provenance"]["direction_counts"]
                for k in (1, 2)]
        assert sorted(bool(w["swapped"]) for w in ways) == [False, True], m
    jev_ways = [json.loads((tier / f"jev-1.14.0@jd_noul_relative_n160d{k}" / "scores_descriptions.json").read_text())["_provenance"]
                for k in (1, 2)]
    assert sorted(bool(p["n_calls_swapped"]) for p in jev_ways) == [False, True]       # Jev's too
    judges = ["gpt-5.6-luna", "claude-haiku-5-5", "gemini-3.1-flash-lite", "jev-1.14.0"]
    weights = tier_judge_weights(judges, "descriptions")      # Jev's description weight (TIER_JUDGE_WEIGHTS_BY_MODE)
    trio_score = {e: float(batch_score(e, True)) for e in ENTITIES}
    jev_score = {e: (SCORES[e] + 3) / 6 * 100 for e in ENTITIES}
    want = combine_tier_judges({**{m: trio_score for m in judges[:3]}, "jev-1.14.0": jev_score}, weights)
    blend = json.loads((tier / four / "scores_descriptions.json").read_text())
    assert blend["result"] == pytest.approx(want, abs=1e-3) and len(blend["_provenance"]["averaged"]["sources"]) == 8
    assert blend["_provenance"]["averaged"]["judge_weights"] == pytest.approx(weights)
    assert all(c["reasoning_effort"] == "none" and c["service_tier"] == "flex" for c in flex.calls)
    usage = json.loads((tier / four / "usage.json").read_text())
    assert set(usage["per_model"]) == {"gpt-5.6-luna:flex", "claude-haiku-5-5:batch", "gemini-3.1-flash-lite:batch",
                                       "jev-1.14.0"}
    # run again: nothing asked, nothing rederived
    n = len(flex.calls)
    assert jd.main(_args(world)) == 0 and len(flex.calls) == n


def test_a_pass_that_did_not_finish_is_not_averaged_and_a_rerun_finishes(world, monkeypatch):
    _in_process(monkeypatch)
    monkeypatch.setattr(ev, "REASK_ROUNDS", 0)
    monkeypatch.setattr(ev, "make_openai_client", lambda service_tier=None: FlakyOpenAI(1))
    extra = ("--judge_model", "gpt-5.6-luna", "--transport", "realtime")
    assert jd.main(_args(world, *extra)) == ev.EXIT_INCOMPLETE
    tier = world.exp / f"{AXES[0][0]}_tier_eval"
    assert not (tier / "gpt-5.6-luna@b10n3dx2").exists()
    monkeypatch.setattr(ev, "make_openai_client", lambda service_tier=None: FakeOpenAI())
    assert jd.main(_args(world, *extra)) == 0
    assert (tier / "gpt-5.6-luna@b10n3dx2" / "usage.json").exists()


def test_one_pass_and_one_judge_write_no_average(world, monkeypatch):
    _in_process(monkeypatch)
    monkeypatch.setattr(ev, "make_openai_client", lambda service_tier=None: FakeOpenAI())
    assert jd.main(_args(world, "--judge_model", "gpt-5.6-luna", "--passes", "1")) == 0
    names = {p.name for p in (world.exp / f"{AXES[0][0]}_tier_eval").iterdir()}
    assert names == {"gpt-5.6-luna@b10n3d1"}


def test_a_dry_run_writes_nothing(world, monkeypatch, capsys):
    _in_process(monkeypatch)
    assert jd.main(_args(world, "--dry_run")) == 0
    assert "Dry run only" in capsys.readouterr().out
    assert not list(world.exp.glob("*_tier_eval"))


@pytest.mark.parametrize("argv", [[], ["--axes_file", "a", "--pair", "x", "y"], ["--pair", "x", "y"], ["--axes_file", "a", "--passes", "0"]])
def test_bad_arguments(argv):
    with pytest.raises(SystemExit):
        jd.main(argv)


def test_a_single_pair(world, monkeypatch):
    _in_process(monkeypatch)
    monkeypatch.setattr(ev, "make_openai_client", lambda service_tier=None: FakeOpenAI())
    pos, neg = AXES[0][1], AXES[0][2]
    argv = ["--pair", pos, neg, "--pair_type", AXES[0][3], "--experiments_dir", str(world.exp), "--instructions_dir", str(world.data),
            "--entities_per_call", "3", "--judge_model", "gpt-5.6-luna", "--requests_per_second", "1000"]
    assert jd.main(argv) == 0
    assert (world.exp / f"{AXES[0][0]}_tier_eval" / "gpt-5.6-luna@b10n3dx2" / "usage.json").exists()
