"""scripts/jev_weight_resilience.py's helpers: the per-entity rules (weighted mean, dropping the outlying score, the
median), the least-agreeing judge, the shortfall summary and the meta-analysis of per-axis gaps."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest
from scipy import stats


def _mod():
    path = Path(__file__).resolve().parents[1] / "jev_weight_resilience.py"
    spec = importlib.util.spec_from_file_location("jev_weight_resilience", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_blend_weights_and_cells():
    m = _mod()
    assert m.blend_w(12).tolist() == [4.0, 3.0, 3.0, 12.0]
    assert m.cells("descriptions", m.JEV, (1, 2)) == ["jev-1.14.0@jd_noul_relative_n160d1", "jev-1.14.0@jd_noul_relative_n160d2"]
    assert m.cells("instructions", "gpt-5.6-luna", (2,)) == [f"gpt-5.6-luna@i1_p0n20d{r}" for r in range(5, 10)]


def test_combine_rule_weights_drops_and_missing():
    m = _mod()
    w = np.array([4.0, 3.0, 3.0, 12.0])
    v = np.array([10.0, 20.0, 30.0, 90.0])
    assert m.combine_rule(v, w) == (pytest.approx((40 + 60 + 90 + 1080) / 22), None)
    # 90 is farthest from the mean (37.5) and from the median (25): dropped, the rest weighted 4:3:3
    assert m.combine_rule(v, w, "mean") == (pytest.approx((40 + 60 + 90) / 10), 3)
    assert m.combine_rule(np.array([0.0, 7.0, 8.0, 9.0]), np.ones(4), "mean")[1] == 0
    # with four scores a < b < c < d, farthest from the mean and farthest from the median are the same score: both
    # drop d exactly when b + c < a + d, so the median rule adds nothing
    rng = np.random.default_rng(4)
    for v4 in rng.normal(size=(500, 4)):
        assert m.combine_rule(v4, np.ones(4), "mean")[1] == m.combine_rule(v4, np.ones(4), "median")[1]
    # with a judge missing nothing is dropped, and the weights renormalise over the judges present
    assert m.combine_rule(np.array([10.0, np.nan, 30.0, 90.0]), w, "mean") == (pytest.approx((40 + 90 + 1080) / 19), None)
    s, i = m.combine_rule(np.full(4, np.nan), w)
    assert np.isnan(s) and i is None


def test_combine_entities_and_median():
    m = _mod()
    per = [{"a": 1.0, "b": 5.0}, {"a": 2.0}, {"a": 3.0, "b": 7.0}, {"a": 50.0, "b": 6.0}]
    s, dropped = m.combine_entities(per, np.ones(4), drop="mean")
    assert s["a"] == pytest.approx(2.0) and dropped["a"] == 3
    assert s["b"] == pytest.approx(6.0) and dropped["b"] is None         # only three judges: nothing dropped
    med, _ = m.combine_entities(per, np.ones(4), median=True)
    assert med == {"a": pytest.approx(2.5), "b": pytest.approx(6.0)}


def test_rank_corr_is_spearman_and_broadcasts():
    m = _mod()
    rng = np.random.default_rng(0)
    x, y = rng.normal(size=50), rng.normal(size=(3, 50))
    got = m.rank_corr(x[None], y)
    assert got == pytest.approx([stats.spearmanr(x, row)[0] for row in y])


def test_least_agreeing_finds_the_odd_judge():
    m = _mod()
    rng = np.random.default_rng(1)
    truth = rng.normal(size=200)
    M = np.array([truth + rng.normal(scale=0.1, size=200) for _ in range(3)] + [rng.normal(size=200)])
    i, agree = m.least_agreeing(M)
    assert i == 3 and agree[3] < min(agree[:3])


def test_shortfall_against_the_better_of_jev_and_trio():
    m = _mod()
    jev, trio, blend = np.array([0.8, 0.6, 0.7]), np.array([0.7, 0.7, 0.7]), np.array([0.78, 0.65, 0.71])
    s = m.shortfall(blend, jev, trio)
    assert s["worst_shortfall"] == pytest.approx(0.05) and s["below_trio"] == 1 and s["worst_vs_trio"] == pytest.approx(-0.05)
    assert s["mean_shortfall"] == pytest.approx((0.02 + 0.05 - 0.01) / 3)     # the third axis beats both: -0.01


def test_meta_analysis_noise_only_and_a_real_spread():
    m = _mod()
    rng = np.random.default_rng(2)
    se = np.full(400, 0.01)
    homogeneous = m.meta_analysis(0.01 + rng.normal(scale=0.01, size=400), se)
    assert homogeneous["tau"] < 0.003 and homogeneous["Q_p"] > 0.001
    spread = m.meta_analysis(0.01 + rng.normal(scale=0.02, size=400) + rng.normal(scale=0.01, size=400), se)
    assert spread["tau"] == pytest.approx(0.02, abs=0.004) and spread["I2"] > 0.7
    # with tau allowed for, the standardised gaps are about N(0, 1); without it they are too wide
    assert np.std(spread["z_tau"]) == pytest.approx(1.0, abs=0.12) and np.std(spread["z_noise"]) > 1.8


def test_tail_summary_counts_and_kurtosis():
    m = _mod()
    rng = np.random.default_rng(3)
    z = np.append(rng.normal(size=58), [-5.0, -4.5, 4.2])
    t = m.tail_summary(z)
    assert t["beyond_3"] >= 3 and t["normal_beyond_3"] == pytest.approx(61 * 2 * stats.norm.sf(3), abs=0.01)
    assert t["excess_kurtosis"] > 1 and t["shapiro_p"] < 0.05
