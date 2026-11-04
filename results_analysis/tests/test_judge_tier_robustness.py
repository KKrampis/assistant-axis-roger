"""Tests for results_analysis/judge_tier_robustness.py on synthetic data.

A tiny activation tree (D=16, 8 slots, 50 layers, a dozen traits and roles,
one trait/role name collision, goal/no-goal marginals) and two axes with
fake judges: legacy gpt/sonnet cells and one tier cell.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.stats import spearmanr

from results_analysis import judge_tier_robustness as jtr
from results_analysis import rho_by_layer as rbl
from results_analysis.canonical_angles.data import build_goal_nogoal_subspaces
from results_analysis.canonical_angles.whitening import (
    DEFAULT_SOFT_K, DEFAULT_SOFT_SHEAR_L, fit_shear, fit_whitening,
)

D, N_SLOTS, N_LAYERS = 16, 8, 50
TRAITS = ["alpha", "beta", "patient"] + [f"t{i}" for i in range(1, 10)]
ROLES = ["r1", "r2", "patient"] + [f"q{i}" for i in range(1, 8)]


def _save_vec(path: Path, arr: np.ndarray, key: str = "vector") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({key: torch.from_numpy(arr.astype(np.float32)).to(torch.bfloat16)}, path)


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory) -> Path:
    d = tmp_path_factory.mktemp("data")
    rng = np.random.default_rng(0)
    default = rng.normal(size=(N_SLOTS, N_LAYERS, D))
    for et, names in (("traits", TRAITS), ("roles", ROLES)):
        _save_vec(d / et / "vectors" / "default.pt", default)
        for n in names:
            _save_vec(d / et / "vectors" / f"{n}.pt", default + rng.normal(size=(N_SLOTS, N_LAYERS, D)))
    _save_vec(d / "combinations" / "vectors" / "default.pt", default)
    for sub in ("r_goal", "r_nogoal", "t_goal", "t_nogoal"):
        for i in range(4):
            _save_vec(d / "combinations/vectors/derived/marginals" / sub / f"x{i}.pt",
                      rng.normal(size=(N_SLOTS, N_LAYERS, D)))
    ax = rng.normal(size=(N_SLOTS, N_LAYERS, D))
    ax /= np.linalg.norm(ax, axis=-1, keepdims=True)
    p = d / "combinations/vectors/derived/axis/theatricality_axis.pt"
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"axis": torch.from_numpy(ax.astype(np.float32)),
                "default_offset": torch.from_numpy(rng.normal(size=(N_SLOTS, N_LAYERS)).astype(np.float32))}, p)
    return d


def _vec(data_dir: Path, et: str, n: str) -> np.ndarray:
    return torch.load(data_dir / et / "vectors" / f"{n}.pt", weights_only=False)["vector"].float().numpy()


def _raw_proj(data_dir: Path, pair_type: str, pos: str, neg: str, slot: int, layer: int) -> dict:
    """Independent raw projection: (v - default) . unit(pos - neg)."""
    default = _vec(data_dir, "traits", "default")[slot, layer]
    u = _vec(data_dir, pair_type, pos)[slot, layer] - _vec(data_dir, pair_type, neg)[slot, layer]
    u = u / np.linalg.norm(u)
    out = {}
    for et, names, tag in (("traits", TRAITS, "T"), ("roles", ROLES, "R")):
        for n in names:
            out[f"{n}|{tag}"] = float((_vec(data_dir, et, n)[slot, layer] - default) @ u)
    return out


@pytest.fixture(scope="module")
def experiments_dir(tmp_path_factory, data_dir) -> Path:
    """Two axes; gpt v2 keys, sonnet v1 bare keys with a gap; a tier judge on one axis."""
    e = tmp_path_factory.mktemp("exp")
    rng = np.random.default_rng(1)
    for axis, pos, neg, pt in (("alpha_vs_beta", "alpha", "beta", "traits"), ("r1_vs_r2", "r1", "r2", "roles")):
        proj = _raw_proj(data_dir, pt, pos, neg, 6, 25)
        sd = np.std(list(proj.values()))
        gpt = {k: int(np.clip(round(v / sd + rng.normal(0, 0.3)), -3, 3)) for k, v in proj.items()
               if k.split("|")[0] not in (pos, neg)}
        son = {}
        for k, v in proj.items():
            name, tag = k.split("|")
            if name in (pos, neg) or (name == "patient" and tag == "R"):
                continue
            son[name] = int(np.clip(round(v / sd + rng.normal(0, 1.0)), -3, 3))
        son["t5"] = None   # a gap ("patient" stays: the collision the lift must drop)
        for sub, scores in (("gpt", gpt), ("sonnet", son)):
            (e / axis / sub).mkdir(parents=True)
            (e / axis / sub / "scores_descriptions.json").write_text(json.dumps({"result": scores}))
        (e / axis / "gpt" / "config.json").write_text(json.dumps({"args": {"pair": [pos, neg], "pair_type": pt}}))
        # cached correlations.json for the gpt cell, as axis_judge_correlation would write it
        corr = jtr.compute_correlations({"descriptions": {k: float(v) for k, v in gpt.items()}},
                                        {6: {k: {"raw": v, "whitened": v} for k, v in proj.items()}},
                                        [6], {pos, neg})
        (e / axis / "gpt" / "correlations.json").write_text(json.dumps(corr))
    tier = e / "alpha_vs_beta_tier_eval" / "fake-judge"
    tier.mkdir(parents=True)
    proj = _raw_proj(data_dir, "traits", "alpha", "beta", 6, 25)
    tier_scores = {k: int(np.clip(round(v), -3, 3)) for k, v in proj.items() if k.split("|")[0] not in ("alpha", "beta")}
    (tier / "scores_descriptions.json").write_text(json.dumps({"result": tier_scores, "_provenance": {}}))
    (tier / "usage.json").write_text(json.dumps({"total_cost_usd": 0.04, "per_model": {}}))
    nousage = e / "alpha_vs_beta_tier_eval" / "no-usage-judge"
    nousage.mkdir(parents=True)
    (nousage / "scores_descriptions.json").write_text(json.dumps({"result": tier_scores}))
    return e


# ---------------------------------------------------------------------------
# Pure functions
# ---------------------------------------------------------------------------

def test_modules_come_from_the_same_tree():
    assert Path(rbl.__file__).resolve().parents[1] == Path(jtr.__file__).resolve().parents[1]


def test_variant_keys():
    vs = jtr.build_variants([0, 6], [25, 49], jtr.REGIMES)
    keys = [v.key for v in vs]
    assert len(keys) == 12 and len(set(keys)) == 12
    assert f"s0_l49_softK{DEFAULT_SOFT_K}" in keys
    assert f"s6_l25_shear{DEFAULT_SOFT_SHEAR_L}" in keys
    assert keys[0] == "s0_l25_raw" and jtr.DEFAULT_VARIANT in keys
    assert jtr.variant_key(6, 25, "soft_K", K=5) == "s6_l25_softK5"
    with pytest.raises(ValueError):
        jtr.variant_key(6, 25, "lw")


def test_cell_rho_excludes_poles_by_bare_name_and_nonfinite():
    scores = {"a|T": 1, "b|T": 2, "c|R": 3, "d|T": 0, "e|R": -1, "pos|T": 3, "pos|R": 3, "neg|T": -3}
    proj = {"a|T": 0.1, "b|T": 0.5, "c|R": 0.9, "d|T": -0.2, "e|R": float("nan"),
            "pos|T": -9.0, "pos|R": 9.0, "neg|T": 5.0}
    rho, n = jtr.cell_rho(scores, proj, ("pos", "neg"))
    assert n == 4
    assert rho == pytest.approx(spearmanr([1, 2, 3, 0], [0.1, 0.5, 0.9, -0.2]).correlation)
    rho2, n2 = jtr.cell_rho({"a|T": 1, "b|T": 2}, {"a|T": 0.0, "b|T": 1.0}, ())
    assert n2 == 2 and math.isnan(rho2)


def test_numeric_and_lifted_scores():
    raw = jtr.numeric_scores({"x": 1, "y": None, "z": True, "w": float("nan"), "patient": 2, "q|R": -1, "v": 0})
    assert raw == {"x": 1.0, "patient": 2.0, "q|R": -1.0, "v": 0.0}
    kinds = {"x": {"traits"}, "patient": {"traits", "roles"}, "q": {"roles"}}
    lifted = jtr.lift_scores(raw, kinds)
    assert lifted == {"x|T": 1.0, "q|R": -1.0}   # collision and orphan dropped


def test_apply_coverage():
    cells = {"a": {"x": 1, "y": 1, "z": 1}, "b": {"x": 1, "y": 1, "z": 1}, "c": {"x": 1, "y": 1}}
    kept, excluded, bar = jtr.apply_coverage(cells)
    assert kept == ["a", "b"] and excluded == {"c": 2} and bar == 3
    kept, excluded, bar = jtr.apply_coverage(cells, min_axes=2)
    assert kept == ["a", "b", "c"] and excluded == {} and bar == 2


def _rho_table(means: dict, axes=("x", "y", "z"), vkey="v") -> dict:
    return {m: {a: {vkey: means[m]} for a in axes} for m in means}


def test_frontier_membership():
    rho = _rho_table({"cheap": 0.5, "mid": 0.6, "dominated": 0.55, "dear": 0.7})
    costs = {"cheap": 0.1, "mid": 0.5, "dominated": 0.8, "dear": 2.0}
    per, frontier = jtr.summarise_variant(rho, costs, list(costs), "v")
    assert frontier == ["cheap", "mid", "dear"]
    assert per["dominated"]["on_frontier"] is False and per["dear"]["on_frontier"] is True
    assert per["mid"]["quality"] == pytest.approx(1 / (1 - 0.6))
    assert per["mid"]["n_axes"] == 3


def test_kendall_tau():
    models = ["a", "b", "c", "d"]
    per = {"ref": {m: {"mean_rho": r} for m, r in zip(models, [0.7, 0.6, 0.5, 0.4])},
           "same": {m: {"mean_rho": r} for m, r in zip(models, [0.9, 0.8, 0.6, 0.1])},
           "rev": {m: {"mean_rho": r} for m, r in zip(models, [0.1, 0.2, 0.3, 0.4])},
           "swap": {m: {"mean_rho": r} for m, r in zip(models, [0.6, 0.7, 0.5, 0.4])}}
    taus = jtr.kendall_tau_vs_default(per, models, "ref")
    assert taus["ref"] == pytest.approx(1.0) and taus["same"] == pytest.approx(1.0)
    assert taus["rev"] == pytest.approx(-1.0)
    assert taus["swap"] == pytest.approx(4 / 6)
    assert jtr.kendall_tau_vs_default({"ref": per["ref"]}, ["a"], "ref") == {"ref": None}


def test_price_bands_and_frontier_counts():
    per = {"cheap": {"mean_rho": 0.5, "mean_cost": 0.1}, "mid": {"mean_rho": 0.6, "mean_cost": 0.2},
           "dear": {"mean_rho": 0.7, "mean_cost": 2.0}}
    b = jtr.price_bands(per, ["cheap", "mid", "dear"], list(per), 0.25)
    assert b == {"cheapest_frontier": "cheap", "best_under_band": "mid", "best_overall": "dear"}
    assert jtr.price_bands(per, ["cheap"], list(per), 0.05)["best_under_band"] is None
    assert jtr.frontier_counts({"v1": ["a", "b"], "v2": ["a"]}, ["a", "b", "c"]) == {"a": 2, "b": 1, "c": 0}


def test_bootstrap_fixed_seed():
    rng = np.random.default_rng(3)
    axes = [f"ax{i:02d}" for i in range(20)]   # the bootstrap sorts the axes
    a = rng.uniform(0.4, 0.8, size=20)
    b = a + rng.normal(0, 0.05, size=20)
    rho = {"top": {x: {"v": float(a[i]), "always": 0.9, "never": 0.1} for i, x in enumerate(axes)},
           "ru": {x: {"v": float(b[i]), "always": 0.5, "never": 0.5} for i, x in enumerate(axes)}}
    r1 = jtr.bootstrap_keep_order(rho, "top", "ru", ["v", "always", "never"], draws=500, seed=7)
    r2 = jtr.bootstrap_keep_order(rho, "top", "ru", ["v", "always", "never"], draws=500, seed=7)
    assert r1 == r2
    assert r1["p_keep_order"]["always"] == 1.0 and r1["p_keep_order"]["never"] == 0.0
    assert 0.0 < r1["p_keep_order"]["v"] < 1.0 and r1["n_axes"] == 20
    idx = np.random.default_rng(7).integers(0, 20, size=(500, 20))
    d = a[idx].mean(axis=1) - b[idx].mean(axis=1)
    assert r1["p_keep_order"]["v"] == pytest.approx(np.mean(d > 0) + 0.5 * np.mean(d == 0))
    r3 = jtr.bootstrap_keep_order(rho, "top", "ru", ["v"], draws=500, seed=8)
    assert r3["p_keep_order"]["v"] != r1["p_keep_order"]["v"] or r3["seed"] == 8


def test_read_axes_tsv(tmp_path):
    p = tmp_path / "axes.tsv"
    p.write_text("# comment\nalpha_vs_beta\talpha\tbeta\ttraits\n\nr1_vs_r2 r1 r2 roles  # trailing\n")
    axes = jtr.read_axes_tsv(p)
    assert axes == [jtr.AxisDef("alpha_vs_beta", "alpha", "beta", "traits"),
                    jtr.AxisDef("r1_vs_r2", "r1", "r2", "roles")]
    p.write_text("a_vs_b a b animals\n")
    with pytest.raises(ValueError):
        jtr.read_axes_tsv(p)


# ---------------------------------------------------------------------------
# Synthetic activation tree
# ---------------------------------------------------------------------------

def test_slices_match_rho_by_layer_caches(data_dir):
    slices = jtr.build_slices(data_dir, [0, 6], [25, 49])
    with jtr._rbl_slots([0, 6]):
        ev_ref, default_ref, _n, kinds = rbl._build_entity_cache(data_dir)
        pool_ref = rbl._build_pool_cache(data_dir, default_ref)
    assert rbl.SLOTS == [0, 3, 6, 7]   # restored
    for si, slot in enumerate([0, 6]):
        for layer in (25, 49):
            ss = slices[(slot, layer)]
            ev = ss.entity_vecs()
            assert set(ev) == set(ev_ref)
            for eid, arr in ev.items():
                np.testing.assert_array_equal(arr[0, 0], ev_ref[eid][si, layer])
            pc = ss.pool_cache()
            for key, arr in pool_ref.items():
                np.testing.assert_array_equal(pc[key][0, 0], arr[si, layer])
            np.testing.assert_array_equal(pc[("combinations", "default")][0, 0], default_ref[si, layer])
    assert jtr.vector_kinds(data_dir) == kinds


def test_projections_per_regime(data_dir):
    slot, layer = 6, 49
    slices = jtr.build_slices(data_dir, [slot], [layer])
    ax = jtr.AxisDef("alpha_vs_beta", "alpha", "beta", "traits")
    variants = jtr.build_variants([slot], [layer], jtr.REGIMES)
    names = [f"t{i}|T" for i in range(1, 10)] + [f"q{i}|R" for i in range(1, 8)] + ["patient|R"]
    proj = jtr.project_variants(slices, [ax], variants, {ax.axis_dir: names}, data_dir)[ax.axis_dir]
    direct = _raw_proj(data_dir, "traits", "alpha", "beta", slot, layer)
    for n in names:
        assert proj[f"s{slot}_l{layer}_raw"][n] == pytest.approx(direct[n], abs=1e-4)
    # soft K: fitted on every trait and role except the two poles, plus default.pt
    default = _vec(data_dir, "traits", "default")[slot, layer]
    pool = [_vec(data_dir, et, n)[slot, layer] for et, ns in (("roles", ROLES), ("traits", TRAITS))
            for n in ns if n not in ("alpha", "beta")] + [default]
    basis = fit_whitening("soft_K", np.stack(pool), K=DEFAULT_SOFT_K)
    u = _vec(data_dir, "traits", "alpha")[slot, layer] - _vec(data_dir, "traits", "beta")[slot, layer]
    u /= np.linalg.norm(u)
    uw = basis.apply(u)
    for n in names:
        name, tag = n.split("|")
        v = _vec(data_dir, "traits" if tag == "T" else "roles", name)[slot, layer] - default
        assert proj[f"s{slot}_l{layer}_softK{DEFAULT_SOFT_K}"][n] == pytest.approx(
            float(basis.apply(v) @ uw) / np.linalg.norm(uw), abs=1e-4)
    # soft shear: fitted on the combined goal / no-goal subspaces
    A_g, A_n = build_goal_nogoal_subspaces(data_dir, slot=slot, layer=layer, kind="combined")
    sb = fit_shear(A_g, A_n, L=DEFAULT_SOFT_SHEAR_L)
    uw = sb.apply(u)
    for n in names:
        name, tag = n.split("|")
        v = _vec(data_dir, "traits" if tag == "T" else "roles", name)[slot, layer] - default
        assert proj[f"s{slot}_l{layer}_shear{DEFAULT_SOFT_SHEAR_L}"][n] == pytest.approx(
            float(sb.apply(v) @ uw) / np.linalg.norm(uw), abs=1e-4)


def test_discover_axes_and_cells(data_dir, experiments_dir):
    kinds = jtr.vector_kinds(data_dir)
    axes = jtr.discover_axes(experiments_dir, kinds)
    assert axes == [jtr.AxisDef("alpha_vs_beta", "alpha", "beta", "traits"),
                    jtr.AxisDef("r1_vs_r2", "r1", "r2", "roles")]
    cells, skipped = jtr.load_cells(axes, experiments_dir, experiments_dir, kinds, "recorded")
    assert sorted(cells) == ["claude-sonnet-4 (May)", "fake-judge", "gpt-4.1-mini (May)"]
    assert skipped == [{"model": "no-usage-judge", "axis": "alpha_vs_beta", "reason": "no usage.json"}]
    son = cells["claude-sonnet-4 (May)"]["alpha_vs_beta"]
    assert "patient|T" not in son.scores and all("|" in k for k in son.scores)   # collision dropped
    # bare file: 10 trait names (patient included) + 9 role names, one of them a gap
    assert son.cost_is_measured is False and son.n_scored == 18
    assert len(son.scores) == 17   # "patient" is a collision name: dropped by the lift
    assert cells["fake-judge"]["alpha_vs_beta"].cost == pytest.approx(0.04)


def _run(tmp_path, data_dir, experiments_dir, *extra, legacy=True):
    out = tmp_path / "out"
    rc = jtr.main(["--data_dir", str(data_dir), "--experiments_dir", str(experiments_dir),
                   "--output_dir", str(out), "--price_basis", "recorded", "--bootstrap_draws", "200",
                   *(["--legacy"] if legacy else []), *extra])
    assert rc == 0
    return json.loads((out / "judge_tier_robustness.json").read_text()), out


def test_end_to_end(tmp_path, data_dir, experiments_dir):
    res, out = _run(tmp_path, data_dir, experiments_dir)
    assert (out / "judge_tier_robustness.png").stat().st_size > 10_000
    assert sorted((out / "judge_tier_robustness_cache").glob("slices_*.npz"))
    r = res["result"]
    assert "_provenance" in res and r["config"]["variants"][6] == "s6_l25_raw"
    assert sorted(r["models"]) == ["claude-sonnet-4 (May)", "gpt-4.1-mini (May)"]
    assert r["excluded_models"] == {"fake-judge": 1}
    assert len(r["rho"]["gpt-4.1-mini (May)"]["alpha_vs_beta"]) == 12
    # the default variant reproduces the cached correlations.json of the gpt cells
    chk = r["check_vs_cached_correlations"]
    assert chk["n_cells_same_n"] == 2 and chk["max_abs_diff_same_n"] < 1e-6
    assert len(r["summary"]["kendall_tau_vs_default"]) == 12
    assert r["summary"]["kendall_tau_vs_default"]["s6_l25_raw"] == pytest.approx(1.0)
    assert set(r["summary"]["frontier_counts"]) == {"claude-sonnet-4 (May)", "gpt-4.1-mini (May)"}
    boot = r["summary"]["bootstrap_keep_order"]
    assert boot["top"] == r["summary"]["default_ranking"][0] and boot["n_axes"] == 2
    # mean rho per variant is the mean of the per-axis rho
    for k in r["config"]["variants"]:
        for m in r["models"]:
            vals = [r["rho"][m][a][k] for a in r["rho"][m]]
            assert r["per_variant"][k]["models"][m]["mean_rho"] == pytest.approx(np.mean(vals))
    # second run reads the cache and gives the same numbers
    res2, _ = _run(tmp_path, data_dir, experiments_dir)
    assert res2["result"]["rho"] == r["rho"]
    # lowering the bar brings in the tier judge (one axis)
    res3, _ = _run(tmp_path, data_dir, experiments_dir, "--min_axes", "1", "--axes", "alpha_vs_beta")
    assert sorted(res3["result"]["models"]) == ["claude-sonnet-4 (May)", "fake-judge", "gpt-4.1-mini (May)"]


def test_legacy_cells_off_by_default(tmp_path, data_dir, experiments_dir):
    """Since 2026-11-01 the May cells enter only with --legacy."""
    res, _ = _run(tmp_path, data_dir, experiments_dir, "--min_axes", "1", "--axes", "alpha_vs_beta", legacy=False)
    assert sorted(res["result"]["models"]) == ["fake-judge"]


def _small_figure_inputs():
    variants = jtr.build_variants([6], [25, 49], ["raw"])
    costs = {"gpt-6-luna@b10x2": 0.02, "gpt-6-luna@b10n20x2": 0.019, "claude-haiku-5-5@b10x2": 0.05}
    per_variant, frontiers = {}, {}
    for shift, v in enumerate(variants):
        rho = {m: {"ax": {v.key: 0.5 + shift + i * 0.01 if shift == 0 else 0.2 + i * 0.01}}
               for i, m in enumerate(costs)}
        per_variant[v.key], frontiers[v.key] = jtr.summarise_variant(rho, costs, list(costs), v.key)
    return variants, list(costs), per_variant, frontiers


def test_figure_options_label_only_the_frontier_with_the_shortest_text_and_each_panel_its_own_y(tmp_path, monkeypatch):
    variants, cells, per_variant, frontiers = _small_figure_inputs()
    seen = []
    monkeypatch.setattr(jtr, "_place_labels", lambda ax, pts, labels, bold, **k: seen.append((labels, bold, ax.get_ylim())))
    opts = jtr.FigureOptions(headline="A headline", cell_labels={c: f"long {i}" for i, c in enumerate(cells)},
                             point_labels={cells[1]: "short"}, series=[[cells[0], cells[1]]],
                             label_frontier_only=True, own_y=True, x_log=True, x_ticks=(0.02, 0.05))
    out = tmp_path / "fig.png"
    jtr.render_figure(out, cells, variants, per_variant, frontiers, {c: True for c in cells}, variants[0].key, 30,
                      "sub", None, opts)
    assert out.stat().st_size > 0
    assert len(seen) == 2
    for (labels, bold, _ylim), k in zip(seen, frontiers):
        assert [bool(t) for t in labels] == [c in frontiers[k] for c in cells] == list(bold)
        assert labels[cells.index("gpt-6-luna@b10n20x2")] == "short"        # the cheapest cell is always on the frontier
    assert seen[0][2] != seen[1][2]                                         # the rows' limits would be equal


def test_figures_share_a_row_by_default(tmp_path, monkeypatch):
    variants, cells, per_variant, frontiers = _small_figure_inputs()
    seen = []
    monkeypatch.setattr(jtr, "_place_labels", lambda ax, pts, labels, bold, **k: seen.append((labels, ax.get_ylim())))
    jtr.render_figure(tmp_path / "fig.png", cells, variants, per_variant, frontiers, {c: True for c in cells},
                      variants[0].key, 30, "sub", None)
    assert seen[0][1] == seen[1][1]
    assert all(all(labels) for labels, _ in seen)


def _grid_script():
    import importlib.util
    path = Path(jtr.__file__).resolve().parents[1] / "scripts" / "judge_tier_grid_12way.py"
    spec = importlib.util.spec_from_file_location("judge_tier_grid_12way", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_grid_figures_cover_the_grid_once_each():
    figs = _grid_script().figures()
    assert set(figs) == {"batch_size", "passes", "reasoning", "all"}
    labels = {stem: f[0] for stem, f in figs.items()}
    assert len(labels["all"]) == 3 * 10 + 9 + 3                  # 10 settings per judge, 9 for Gemini (up to 3 passes), three blends
    assert len(set(labels["all"].values())) == len(labels["all"])
    assert len(labels["batch_size"]) == 4 * 5 and len(labels["passes"]) == 3 * 4 + 3 and len(labels["reasoning"]) == 4 * 3
    for stem in ("batch_size", "passes", "reasoning"):
        assert set(labels[stem]) <= set(labels["all"])
        for run in figs[stem][2]:                                # each arrow chain lies inside its figure
            assert set(run) <= set(labels[stem])
    # N=10, 2 passes, no reasoning is one cell, in all three dimensions
    assert "gpt-6-luna@b10x2" in labels["batch_size"] and "gpt-6-luna@b10x2" in labels["passes"]
    assert "gpt-6-luna@b10x2" in labels["reasoning"] and labels["all"]["gpt-6-luna@b10x2"] == "GPT-6 base"


def test_grid_script_recomputes_each_figures_frontier(tmp_path, monkeypatch):
    mod = _grid_script()
    cells = list(mod.figures()["all"][0])
    variants = jtr.build_variants([0, 6], [25, 49], list(jtr.REGIMES))
    rng = np.random.default_rng(1)
    payload = {"config": {"slots": [0, 6], "layers": [25, 49], "regimes": list(jtr.REGIMES), "soft_K": 2, "shear_L": 3,
                          "default_variant": "s6_l25_raw"},
               "axes": [{"axis_dir": "ax"}], "models": {c: {"mean_cost": float(rng.uniform(0.01, 0.1))} for c in cells},
               "rho": {c: {"ax": {v.key: float(rng.uniform(0.4, 0.7)) for v in variants}} for c in cells}}
    src = tmp_path / "rho.json"
    src.write_text(json.dumps(payload))
    drawn = {}
    monkeypatch.setattr(mod.rob, "render_figure", lambda out, order, vs, pv, fr, *a, **k: drawn.setdefault(Path(out).stem, fr))
    monkeypatch.setattr("sys.argv", ["x", "--robustness_json", str(src), "--output_dir", str(tmp_path)])
    mod.main()
    assert set(drawn) == {f"judge_tier_grid_12way_{s}" for s in ("batch_size", "passes", "reasoning", "all")}
    for stem in ("batch_size", "passes", "reasoning"):
        labels = mod.figures()[stem][0]
        assert all(set(f) <= set(labels) for f in drawn[f"judge_tier_grid_12way_{stem}"].values())
    res = json.loads((tmp_path / "judge_tier_grid_12way.json").read_text())
    res = res.get("result", res)
    assert all(0 <= c["on_frontier_of_12"] <= 12 for fig in ("batch_size", "all") for c in res[fig]["cells"].values())


# ---------------------------------------------------------------------------
# The other figure scripts: tracked producers that record their inputs
# ---------------------------------------------------------------------------

def _script(name):
    import importlib.util
    path = Path(jtr.__file__).resolve().parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _grid_rho_table(cells):
    variants = jtr.build_variants([0, 6], [25, 49], list(jtr.REGIMES))
    rng = np.random.default_rng(1)
    axes = [f"ax{i}" for i in range(4)]
    return {"config": {"slots": [0, 6], "layers": [25, 49], "regimes": list(jtr.REGIMES), "soft_K": 2, "shear_L": 3,
                       "default_variant": "s6_l25_raw"},
            "axes": [{"axis_dir": a} for a in axes],
            "models": {c: {"mean_cost": float(rng.uniform(0.01, 0.1))} for c in cells},
            "rho": {c: {a: {v.key: float(rng.uniform(0.4, 0.7)) for v in variants} for a in axes} for c in cells}}


def _png_inputs(path):
    from PIL import Image
    return Image.open(path).info.get("Inputs")


def test_grid_shapes_writes_both_figures_with_inputs(tmp_path, monkeypatch):
    mod = _script("judge_tier_grid_shapes")
    cells = [c for name, base, plain, first, k in mod.JUDGES
             for dim in mod.dimension_arms(base, plain, first, min(k, 3)).values() for c in dim.values()]
    src = tmp_path / "rho.json"
    src.write_text(json.dumps(_grid_rho_table(sorted(set(cells)))))
    monkeypatch.setattr("sys.argv", ["x", "--robustness_json", str(src), "--output_dir", str(tmp_path), "--cache-policy", "off"])
    mod.main()
    assert _png_inputs(tmp_path / "judge_tier_grid_shapes.png") and _png_inputs(tmp_path / "judge_tier_grid_own_frontier.png")
    res = json.loads((tmp_path / "judge_tier_grid_shapes.json").read_text())
    assert set(res["_provenance"]) and set(res["result"]["own_frontier"]) == {"Luna", "GPT-6", "Haiku", "Gemini"}
    assert res["result"]["shapes"]["Gemini"]["passes"]["2x"]["drho"] == 0.0     # the judge's own base


def test_tuning_panels_writes_a_figure_with_inputs(tmp_path):
    mod = _script("judge_tier_tuning_panels")
    cells = [c for j in mod.JUDGES for c in (j, f"{j}@c8", f"{j}@b10x2")]
    payload = _grid_rho_table(cells)
    payload["per_variant"] = {"s6_l25_raw": {"models": {c: {"mean_rho": 0.5 + 0.01 * i} for i, c in enumerate(cells)}}}
    src, out = tmp_path / "rho.json", tmp_path / "tuning.png"
    src.write_text(json.dumps(payload))
    mod.main(["--robustness_json", str(src), "--output", str(out), "--cache-policy", "off"])
    assert _png_inputs(out)


def test_grid_analysis_records_its_inputs(tmp_path, monkeypatch):
    mod = _script("judge_tier_grid_analysis")
    E = tmp_path / "exp"
    axes = tmp_path / "axes.tsv"
    axes.write_text("a_vs_b\ta\tb\ttraits\n")
    arms = {"gpt-6-luna@b10s1": 0.60, "gpt-6-luna@b10x2": 0.65, "gpt-6-luna@b10n20x2": 0.66}
    for cell, rho in arms.items():
        d = E / "a_vs_b_tier_eval" / cell
        d.mkdir(parents=True)
        (d / "correlations.json").write_text(json.dumps({"descriptions": {"6": {"raw": {"rho": rho, "n": 9}}}}))
        (d / "usage.json").write_text(json.dumps({"total_cost_usd": 0.1, "per_model": {}}))
    monkeypatch.setattr("sys.argv", ["x", "--experiments_dir", str(E), "--axes_file", str(axes), "--output_dir", str(tmp_path)])
    mod.main()
    assert _png_inputs(tmp_path / "judge_tier_grid_dev30.png")
    res = json.loads((tmp_path / "judge_tier_grid_dev30.json").read_text())
    assert res["_provenance"]["inputs"] and {r["cell"] for r in res["result"]} == set(arms)


# ---------------------------------------------------------------------------
# scripts/judge_instruction_12way.py: Nx = N passes over all five instructions (Roger, 2026-11-03),
# averaged over the choices of passes
# ---------------------------------------------------------------------------

def test_instruction_12way_arms_are_every_judge_set_at_1x_and_2x_luna_to_4x_and_the_mixes():
    mod = _script("judge_instruction_12way")
    # the figure before 2026-11-04: no Jev, the axes' own order (--without_jev --directions own)
    mod.WITH_JEV, mod.BALANCED = False, False
    arms = mod.arms()
    assert len(arms) == 7 * 2 + 2 + 4 and len({lab for lab, _ in arms.values()}) == 20
    assert arms["gpt-5.6-luna+claude-haiku-5-5@i1_p0n20x2"] == ("Luna + Haiku 2x", {"gpt-5.6-luna": 2, "claude-haiku-5-5": 2})
    assert arms["gpt-5.6-luna@i1_p0n20x4"] == ("Luna 4x", {"gpt-5.6-luna": 4})          # Luna alone (Roger, 2026-11-03)
    assert not any(k.endswith(("x3", "x4")) for k in arms if not k.startswith("gpt-5.6-luna@"))
    assert arms["gpt-5.6-luna+claude-haiku-5-5+gemini-3.1-flash-lite@i1_p0n20m2-1-1"] == (
        "Luna 2x + Haiku 1x + Gemini 1x", {"gpt-5.6-luna": 2, "claude-haiku-5-5": 1, "gemini-3.1-flash-lite": 1})
    assert arms["gpt-5.6-luna+claude-haiku-5-5+gemini-3.1-flash-lite@i1_p0n20m2-2-1"][0] == "Luna 2x + Haiku 2x + Gemini 1x"
    assert len(mod.replicates({"gpt-5.6-luna": 2, "claude-haiku-5-5": 2, "gemini-3.1-flash-lite": 1})) == 6 * 1 * 2
    series = mod.series()
    luna = next(s for s in series if s[0].startswith("gpt-5.6-luna@"))
    assert [c[-2:] for c in luna] == ["x1", "x2", "x3", "x4"]
    assert len(series) == 7 and all([c[-2:] for c in s] == ["x1", "x2"] for s in series if s is not luna)
    assert sorted(c for s in series for c in s) == sorted(k for k, (_, spec) in arms.items() if len(set(spec.values())) == 1)
    # every arm fits the passes judged
    assert all(k <= mod.PASSES[m] for _, spec in arms.values() for m, k in spec.items())
    # the defaults since 2026-11-04 (Roger: Jev beside the trio, balanced directions): the six Jev arms, three more
    # series, and a 2x choice of passes one each way (Luna 2x of 4: 4 choices, not 6)
    mod.WITH_JEV, mod.BALANCED = True, True
    arms = mod.arms()
    assert len(arms) == 7 * 2 + 2 + 4 + 6 and len({lab for lab, _ in arms.values()}) == 26
    assert arms["jev-1.14.0@i1_p0n20x2"] == ("Jev 2x", {"jev-1.14.0": 2})
    assert len(mod.series()) == 10
    assert len(mod.replicates({"gpt-5.6-luna": 2, "claude-haiku-5-5": 2, "gemini-3.1-flash-lite": 1})) == 4 * 1 * 2
    assert all(k <= mod.passes_of(mod.INSTRUCTIONS)[m] for _, spec in arms.values() for m, k in spec.items())


def test_instruction_12way_a_pass_is_all_five_instructions_and_choices_are_per_judge():
    mod = _script("judge_instruction_12way")
    assert list(mod.pass_rounds(0)) == [0, 1, 2, 3, 4] and list(mod.pass_rounds(3)) == [15, 16, 17, 18, 19]
    passes = {"a": 4, "b": 2}
    mod.BALANCED = False            # the axes' own order: every choice of passes
    assert len(mod.replicates({"a": 1}, passes)) == 4 and len(mod.replicates({"a": 2}, passes)) == 6
    assert len(mod.replicates({"a": 4}, passes)) == 1
    mix = mod.replicates({"a": 2, "b": 1}, passes)
    assert len(mix) == 6 * 2 and all(len(rep["a"]) == 10 and len(rep["b"]) == 5 for rep in mix)
    assert {rep["b"] for rep in mix} == {(0, 1, 2, 3, 4), (5, 6, 7, 8, 9)}
    mod.BALANCED = True             # balanced (the default since 2026-11-04): an even count takes one pass each way
    assert len(mod.replicates({"a": 1}, passes)) == 4 and len(mod.replicates({"a": 2}, passes)) == 4
    assert len(mod.replicates({"a": 4}, passes)) == 1 and len(mod.replicates({"a": 2, "b": 1}, passes)) == 4 * 2
    # Luna's two judgements average to 2, Haiku says 6: per judge, then Luna 4 to Haiku 3 (Roger, 2026-11-03), not
    # (1 + 3 + 6) / 3; an entity Haiku left out is Luna's alone; two judges without Luna weigh alike
    assert mod.combine({"gpt-5.6-luna": [{"e": 1, "f": 5}, {"e": 3}], "claude-haiku-5-5": [{"e": 6}]}) == pytest.approx(
        {"e": (4 * 2 + 3 * 6) / 7, "f": 5.0})
    assert mod.combine({"a": [{"e": 1}, {"e": 3}], "b": [{"e": 6}]}) == {"e": 4.0}


def test_instruction_12way_table_costs_whole_passes_and_averages_over_the_choices():
    mod = _script("judge_instruction_12way")
    rng = np.random.default_rng(3)
    names = [f"t{i}" for i in range(40)]
    proj = {n: float(rng.normal()) for n in names}
    ax = jtr.AxisDef("x_vs_y", "x", "y", "traits")
    vkeys = ["v1", "v2"]
    projections = {"x_vs_y": {"v1": proj, "v2": {n: -p for n, p in proj.items()}}}
    scores, costs = {}, {}
    for r in range(10):
        scores["x_vs_y", "good", r] = dict(proj)                                    # agrees with v1 exactly
        scores["x_vs_y", "noise", r] = {n: float(rng.normal()) for n in names}
        costs["x_vs_y", "good", r], costs["x_vs_y", "noise", r] = 0.01 * (r + 1), 0.02
    passes = {"good": 2, "noise": 2}
    rho, cost, per_choice = mod.arm_table({"good@x1": {"good": 1}, "good+noise@x2": {"good": 2, "noise": 2}},
                                          [ax], scores, costs, projections, vkeys, passes=passes)
    assert rho["good@x1"]["x_vs_y"]["v1"] == pytest.approx(1.0) and rho["good@x1"]["x_vs_y"]["v2"] == pytest.approx(-1.0)
    assert cost["good@x1"] == pytest.approx((0.15 + 0.40) / 2)                      # a whole pass, either of the two
    assert cost["good+noise@x2"] == pytest.approx(0.55 + 10 * 0.02)                 # both passes of each judge
    assert len(per_choice["good@x1"]["v1"]) == 2 and len(per_choice["good+noise@x2"]["v1"]) == 1
    assert 0 < rho["good+noise@x2"]["x_vs_y"]["v1"] < 1


def test_instruction_12way_fits_the_mix_weights_and_checks_them_out_of_sample():
    mod = _script("judge_instruction_12way")
    rng = np.random.default_rng(5)
    axes = [jtr.AxisDef(f"a{i}_vs_b{i}", f"a{i}", f"b{i}", "traits") for i in range(6)]
    scores, projection = {}, {}
    for ax in axes:
        names = [f"t{i}|T" for i in range(60)]
        proj = {n: float(rng.normal()) for n in names}
        projection[ax.axis_dir] = proj
        for r in range(10):       # "good" tracks the projection with a little noise, "noise" does not
            scores[ax.axis_dir, "good", r] = {n: p + 0.3 * float(rng.normal()) for n, p in proj.items()}
            scores[ax.axis_dir, "noise", r] = {n: float(rng.normal()) for n in names}
    spec, passes = {"good": 2, "noise": 1}, {"good": 2, "noise": 2}
    grid = mod.weight_grid(2)
    assert len(grid) == 21 and np.allclose(grid.sum(1), 1)
    assert len(mod.weight_grid(3)) == 231
    G = mod.fit_rho(spec, axes, scores, projection, grid, passes=passes)
    fit = mod.fit_weights(G, grid, np.array([True, False] * 3), list(spec))
    assert fit["weights"]["good"] >= 0.9 and fit["in_sample_gain"] > 0
    assert set(fit["split_half_weights"]) == {"dev", "held"} and fit["leave_one_axis_out_gain"] > 0
    # the fitted weights reach the table, the others keep the default
    projections = {a: {"v": p} for a, p in projection.items()}
    costs = {k: 0.01 for k in scores}
    rho_fit, _, _ = mod.arm_table({"m": spec}, axes, scores, costs, projections, ["v"], passes=passes,
                                  weights={"m": fit["weights"]})
    rho_def, _, _ = mod.arm_table({"m": spec}, axes, scores, costs, projections, ["v"], passes=passes)
    assert np.mean([r["v"] for r in rho_fit["m"].values()]) > np.mean([r["v"] for r in rho_def["m"].values()])


def test_instruction_12way_descriptions_mode_reads_the_jl_cells_two_passes_each():
    mod = _script("judge_instruction_12way")
    d = mod.MODES["descriptions"]
    assert d.cell("gpt-5.6-luna", 0) == "gpt-5.6-luna@none@b10n20s1jl" and d.cell("claude-haiku-5-5", 1) == "claude-haiku-5-5@b10n20s2jl"
    assert d.rounds_per_pass == 1 and d.passes == {"gpt-5.6-luna": 4, "claude-haiku-5-5": 2, "gemini-3.1-flash-lite": 2}
    mod.WITH_JEV, mod.BALANCED = False, False       # the figure before 2026-11-04
    arms = mod.arms(d)
    assert len(arms) == 7 * 2 + 2 + 4 and all(k <= d.passes[m] for _, spec in arms.values() for m, k in spec.items())
    assert "gpt-5.6-luna@b10n20jlx4" in arms and "claude-haiku-5-5@b10n20jlx3" not in arms
    assert len(mod.replicates({"gpt-5.6-luna": 2, "claude-haiku-5-5": 1}, d.passes, d.rounds_per_pass)) == 6 * 2
    assert mod.replicates({"a": 1}, {"a": 2}, 1) == [{"a": (0,)}, {"a": (1,)}]
    assert sorted(len(s) for s in mod.series(d)) == [2] * 6 + [4]
    assert mod.INSTRUCTIONS.cell("gpt-5.6-luna", 7) == "gpt-5.6-luna@i1_p0n20r7"
    mod.WITH_JEV, mod.BALANCED = True, True         # the defaults since 2026-11-04: with Jev, balanced
    assert len(mod.arms(d)) == 26 and sorted(len(s) for s in mod.series(d)) == [2] * 9 + [4]
    assert len(mod.replicates({"gpt-5.6-luna": 2, "claude-haiku-5-5": 1}, d.passes, d.rounds_per_pass)) == 4 * 2
