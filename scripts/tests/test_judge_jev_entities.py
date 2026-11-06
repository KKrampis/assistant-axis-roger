"""scripts/judge_jev_entities.py's pairwise design (2026-11-04): the pairs a batch asks about, and the win shares
read from their answers.  The whole-run tests, on judge_tier_cost_eval's fixture, are the jev_entities tests in
results_analysis/tests/test_judge_tier_cost_eval.py."""
import importlib.util
from pathlib import Path

import pytest


def _mod():
    path = Path(__file__).resolve().parents[1] / "judge_jev_entities.py"
    spec = importlib.util.spec_from_file_location("judge_jev_entities", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_pair_list_every_pair_or_a_ring():
    mod = _mod()
    assert mod.pair_list(4, None) == [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    ring = mod.pair_list(10, 2)
    assert len(ring) == 20 and all(a < b for a, b in ring)
    counts = [sum(i in p for p in ring) for i in range(10)]
    assert counts == [4] * 10                       # each entry against its two neighbours either side
    assert mod.pair_list(5, 2) == mod.pair_list(5, None)     # a ring that covers everything is every pair


def test_pair_wins_are_expected_shares_of_wins():
    mod = _mod()
    # entry 1 beats 2 surely, ties 3; entry 2 loses to 3 surely
    answers = {"p1_2": {"noul": 1.0}, "p1_3": {"noul": 0.5}, "p2_3": {"noul": 0.0}}
    assert mod.pair_wins(answers, 3) == [75.0, 0.0, 75.0]
    assert mod.pair_wins({"p1_2": {"noul": 1.5}, "e1": {"noul": 0.2}}, 2) == [None, None]   # out of range; not a pair
    assert mod.pair_wins({}, 2) == [None, None]


def test_pairs_questions_name_both_entries(monkeypatch):
    mod = _mod()
    from types import SimpleNamespace
    spec = SimpleNamespace(neg_examples=["harsh"], pos_examples=["gentle"], neg_pole="cruel", pos_pole="kind")
    show = lambda n, etype: n
    group = [("traits", "calm", "The calm one."), ("roles", "pilot", "The pilot one."), ("traits", "bold", "The bold one.")]
    monkeypatch.setattr(mod, "PAIR_RING", None)
    state, qs = mod.wording_base(spec, "traits", group, "descriptions", show, "pairs", 2)
    assert sorted(qs) == ["p1_2", "p1_3", "p2_3"] and len(state["entries"]) == 3
    q = qs["p1_2"]
    assert q["type"] == "noul" and "entry 1, calm (trait)" in q["instructions"] and "entry 2, pilot (role)" in q["instructions"]
    assert "nearer the gentle end" in q["instructions"]


def test_directions_balance_like_the_trio():
    """Roger's rule (2026-11-04): descriptions, the two passes opposite on every axis; instructions, Jev's column
    with the trio's judges each of the first four instruction positions 2:2, and Jev itself 3:2 or 2:3."""
    mod = _mod()
    ev = mod.ev
    axes = [f"axis_{i}_vs_other_{i}" for i in range(40)]
    for a in axes:
        assert mod.call_swapped("descriptions", a, 0, None) != mod.call_swapped("descriptions", a, 1, None)
        assert not mod.call_swapped("own", a, 0, None) and mod.call_swapped("all", a, 0, None)
    from types import SimpleNamespace
    old = (ev.DIRECTIONS, ev.INSTRUCTION_ROUND)
    try:
        ev.DIRECTIONS = "instructions"
        for a in axes:
            job = SimpleNamespace(axis_dir=a)
            split = []
            for r in range(5):
                ev.INSTRUCTION_ROUND = r
                trio = [ev.group_swapped(job, ["x"], m) for m in ev.INSTRUCTION_DIRECTION_JUDGES]
                split.append(sum(trio) + mod.call_swapped("instructions", a, 0, r))
            assert sorted(split).count(2) >= 4, (a, split)
            jev = sum(mod.call_swapped("instructions", a, 0, r) for r in range(5))
            assert jev in (2, 3)
            # a second pass (rounds 5-9) is the first reversed
            assert all(mod.call_swapped("instructions", a, 0, r) != mod.call_swapped("instructions", a, 0, r + 5)
                       for r in range(5))
    finally:
        ev.DIRECTIONS, ev.INSTRUCTION_ROUND = old


def test_strong_and_relative_wordings(monkeypatch):
    mod = _mod()
    from types import SimpleNamespace
    spec = SimpleNamespace(neg_examples=["harsh"], pos_examples=["gentle"], neg_pole="cruel", pos_pole="kind")
    group = [("traits", "calm", "The calm one."), ("roles", "pilot", "The pilot one.")]
    show = lambda n, etype: n
    state, qs = mod.wording_strong(spec, "traits", group, "descriptions", show, "noul", 2)
    assert sorted(qs) == ["e1", "e1n", "e1s", "e2", "e2n", "e2s"]
    assert "strongly at the gentle end" in qs["e1s"]["instructions"] and "strongly at the harsh end" in qs["e1n"]["instructions"]
    ans = {"e1": {"noul": 0.9}, "e1s": {"noul": 0.6}, "e1n": {"noul": 0.0}}
    assert mod.strong_value(ans, 1) == pytest.approx(100 * (0.9 + 0.6 + 1.0) / 3, abs=1e-3)
    assert mod.strong_value({"e1": {"noul": 0.9}}, 1) is None
    _, qr = mod.wording_relative(spec, "traits", group, "descriptions", show, "noul", 2)
    assert "than most of the other entries" in qr["e2"]["instructions"] and qr["e2"]["type"] == "noul"
    with pytest.raises(SystemExit):
        mod.wording_strong(spec, "traits", group, "descriptions", show, "score", 5)
