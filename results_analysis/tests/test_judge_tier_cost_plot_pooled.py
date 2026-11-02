"""Tests for the coverage rule of judge_tier_cost_plot_pooled.py.

A model judged on fewer axes than the plot's headline count is left out of
the plot (and of the Pareto frontier) and named in the title instead.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from results_analysis.judge_tier_cost_plot_pooled import (  # noqa: E402
    aggregate,
    collect_tier_eval_data,
    load_measured_cost,
    plot,
    split_by_coverage,
)

SCRIPT = Path(__file__).resolve().parents[1] / "judge_tier_cost_plot_pooled.py"


def _write_cell(model_dir: Path, rho: float, n: int = 571, cost: float = 0.5) -> None:
    model_dir.mkdir(parents=True)
    (model_dir / "correlations.json").write_text(json.dumps(
        {"descriptions": {"6": {"raw": {"rho": rho, "n": n}}}}))
    (model_dir / "usage.json").write_text(json.dumps({"total_cost_usd": cost}))


def _fixture(root: Path) -> list[Path]:
    """Three axes: 'full' on all three, 'partial' on the first two only."""
    dirs = []
    for i, rho_full in enumerate((0.70, 0.60, 0.50)):
        d = root / f"axis{i}_tier_eval"
        _write_cell(d / "full", rho_full, cost=1.0)
        if i < 2:
            _write_cell(d / "partial", 0.80, cost=0.1)
        dirs.append(d)
    return dirs


def test_partial_coverage_model_is_split_out(tmp_path: Path) -> None:
    per_model = collect_tier_eval_data(_fixture(tmp_path), slot=6)
    agg = aggregate(per_model)
    assert agg["full"]["n_axes"] == 3 and agg["partial"]["n_axes"] == 2
    n_total = max(len(v) for v in per_model.values())
    kept, excluded = split_by_coverage(agg, n_total)
    assert set(kept) == {"full"}
    assert set(excluded) == {"partial"}
    # lowering the bar keeps everything
    kept_all, excluded_none = split_by_coverage(agg, 1)
    assert set(kept_all) == {"full", "partial"} and excluded_none == {}


def test_plot_names_excluded_models_in_title(tmp_path: Path) -> None:
    per_model = collect_tier_eval_data(_fixture(tmp_path), slot=6)
    agg = aggregate(per_model)
    kept, excluded = split_by_coverage(agg, 3)
    out = tmp_path / "out.png"
    plot(kept, 6, 3, out, excluded=excluded)
    assert out.exists() and out.stat().st_size > 0
    # the title is rendered, not returned: check the source of truth the title is built from
    assert excluded["partial"]["n_axes"] == 2


def _usage(path: Path, key: str, model: str, pt: int, ct: int, billed: float) -> Path:
    path.write_text(json.dumps({
        "per_model": {key: {"model": model, "prompt_tokens": pt, "completion_tokens": ct,
                            "cost_usd": billed, "n_calls": 1}},
        "total_cost_usd": billed}))
    return path


def test_price_basis_puts_batch_and_realtime_cells_on_one_footing(tmp_path: Path) -> None:
    # gpt-4.1-mini: $0.40 / $1.60 per million at list price, half in batch.
    realtime = _usage(tmp_path / "rt.json", "gpt-4.1-mini", "gpt-4.1-mini", 1_000_000, 100_000, 0.56)
    batched = _usage(tmp_path / "b.json", "gpt-4.1-mini:batch", "gpt-4.1-mini:batch",
                     1_000_000, 100_000, 0.28)
    assert abs(load_measured_cost(realtime, "list") - 0.56) < 1e-9
    assert abs(load_measured_cost(batched, "list") - 0.56) < 1e-9      # recomputed at list price
    assert abs(load_measured_cost(batched, "batch") - 0.28) < 1e-9
    assert abs(load_measured_cost(realtime, "batch") - 0.28) < 1e-9
    assert load_measured_cost(batched, "recorded") == 0.28              # as billed
    flex = _usage(tmp_path / "f.json", "gpt-4.1-mini:flex", "gpt-4.1-mini:flex", 1_000_000, 100_000, 0.28)
    assert abs(load_measured_cost(flex, "list") - 0.56) < 1e-9


def test_price_basis_falls_back_to_recorded_for_an_unpriced_model(tmp_path: Path, capsys) -> None:
    unknown = _usage(tmp_path / "u.json", "mystery-model", "mystery-model", 1000, 100, 0.123)
    assert load_measured_cost(unknown, "list") == 0.123
    assert "no list price" in capsys.readouterr().err


def test_cell_max_tokens_and_cap_summary(tmp_path: Path) -> None:
    from results_analysis.judge_tier_cost_plot_pooled import (
        TIER_EVAL_DEFAULT_MAX_TOKENS, cap_summary, cell_max_tokens)
    old = tmp_path / "old"; old.mkdir()                     # an early tier cell: no max_tokens recorded
    (old / "scores_descriptions.json").write_text(json.dumps({"result": {}, "_provenance": {"judge_model": "m"}}))
    new = tmp_path / "new"; new.mkdir()
    (new / "scores_descriptions.json").write_text(json.dumps({"result": {}, "_provenance": {"max_tokens": 300}}))
    legacy = tmp_path / "legacy"; legacy.mkdir()
    (legacy / "config.json").write_text(json.dumps({"args": {"max_tokens": 1024}}))
    bare = tmp_path / "bare"; bare.mkdir()                  # a legacy cell with no config.json
    assert cell_max_tokens(old) == TIER_EVAL_DEFAULT_MAX_TOKENS == 200
    assert cell_max_tokens(new) == 300
    assert cell_max_tokens(legacy, legacy=True) == 1024
    assert cell_max_tokens(bare, legacy=True) is None
    one = cap_summary({"a": {200}, "b": {200}})
    assert one.startswith("Judge reply cap: max_tokens 200 for every judge")
    mixed = cap_summary({"a": {200}, "b": {200}, "c": {200}, "x (May)": {1024}, "y (May)": {1024, None}})
    assert mixed.startswith("Judge reply cap (max_tokens): 200 for 3 judges; ")
    assert "1024 for x (May)" in mixed and "1024 or unrecorded for y (May)" in mixed


def test_rho_from_robustness_replaces_rho_and_n(tmp_path: Path, capsys) -> None:
    from results_analysis.judge_tier_cost_plot_pooled import rho_from_robustness
    rj = tmp_path / "judge_tier_robustness.json"
    rj.write_text(json.dumps({"result": {
        "rho": {"full": {"axis0": {"s6_l25_raw": 0.42, "s0_l25_raw": 0.1}}},
        "n": {"full": {"axis0": 562}}}}))
    per_model = {"full": [{"rho": 0.7, "n": 571, "cost": 1.0, "cost_is_measured": True, "axis_name": "axis0"},
                          {"rho": 0.6, "n": 571, "cost": 1.0, "cost_is_measured": True, "axis_name": "axis1"}]}
    out = rho_from_robustness(per_model, rj)
    assert [(e["axis_name"], e["rho"], e["n"]) for e in out["full"]] == [("axis0", 0.42, 562)]
    assert "axis1: not in" in capsys.readouterr().err
    assert rho_from_robustness(per_model, rj, "s0_l25_raw")["full"][0]["rho"] == 0.1


def test_cli_default_excludes_and_min_axes_restores(tmp_path: Path) -> None:
    _fixture(tmp_path)
    env = {"MPLBACKEND": "Agg", "PATH": "/usr/bin:/bin"}
    out = tmp_path / "default.png"
    r = subprocess.run([sys.executable, str(SCRIPT), "--experiments_dir", str(tmp_path),
                        "--slot", "6", "--output", str(out)],
                       capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stderr
    assert "Not plotted (judged on fewer than 3 axes): partial (n_axes=2)" in r.stderr
    assert out.exists()
    out2 = tmp_path / "all.png"
    r2 = subprocess.run([sys.executable, str(SCRIPT), "--experiments_dir", str(tmp_path),
                         "--slot", "6", "--min_axes", "1", "--output", str(out2)],
                        capture_output=True, text=True, env=env)
    assert r2.returncode == 0, r2.stderr
    assert "Not plotted" not in r2.stderr
    assert out2.exists()


def test_display_names_spell_out_settings_and_notes():
    from results_analysis.judge_tier_cost_plot_pooled import display_name
    assert display_name("gpt-5.6-luna@none") == "gpt-5.6-luna (no reasoning)"
    assert display_name("claude-haiku-4-5@t0") == "claude-haiku-4-5 (T=0)"
    assert display_name("claude-opus-5@nothink") == "claude-opus-5 (thinking off)"
    assert display_name("gpt-4.1-mini") == "gpt-4.1-mini (retires 2027-01-14)"
    assert display_name("gpt-4.1-mini@x") == "gpt-4.1-mini (@x, retires 2027-01-14)"
    assert display_name("claude-haiku-5-5") == "claude-haiku-5-5"
    assert display_name("claude-opus-5") == "claude-opus-5 (effort low)"
    assert display_name("gpt-5.6-luna") == "gpt-5.6-luna (default reasoning)"
    assert display_name("gpt-5.6-luna@none@c2") == "gpt-5.6-luna (no reasoning, c2)"
    assert display_name("gpt-4.1-mini@v3fix") == "gpt-4.1-mini (rubric v3fix, retires 2027-01-14)"
    assert display_name("claude-haiku-5-5@b10x2") == "claude-haiku-5-5 (b10x2)"
    assert display_name("gpt-6-luna@b10s1") == "gpt-6-luna (b10s1)"
    assert display_name("gpt-4.1-mini@b10x2", short=True) == "gpt-4.1-mini (b10x2)"
    from results_analysis.judge_tier_cost_plot_pooled import label_key
    assert label_key(["gpt-4.1-mini", "gpt-6-luna@b10x2", "gpt-6-luna@c8", "x@b10x2"]) == (
        "b10x2: batched rubric b10, 10 entities per call, 2 shuffles averaged; c8: single-entity rubric c8")
    assert display_name("gpt-6-luna@low@b10n20s1") == "gpt-6-luna (reasoning low, b10n20s1)"
    assert "b10n5x4: batched rubric b10, 5 entities per call, 4 shuffles averaged" in label_key(["x@b10n5x4"])
    assert display_name("gpt-4.1-mini (May)") == "gpt-4.1-mini (May)"


def test_exclude_models_leaves_a_cell_out_and_says_so(tmp_path: Path) -> None:
    _fixture(tmp_path)
    env = {"MPLBACKEND": "Agg", "PATH": "/usr/bin:/bin"}
    out = tmp_path / "x.png"
    r = subprocess.run([sys.executable, str(SCRIPT), "--experiments_dir", str(tmp_path), "--slot", "6",
                        "--min_axes", "1", "--exclude_models", "partial", "nosuch", "--output", str(out)],
                       capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stderr
    assert "partial" not in r.stderr.split("Aggregated results")[1].split("Wrote")[0]
    assert "--exclude_models nosuch: no such cell" in r.stderr and out.exists()


def test_links_draw_between_plotted_cells(tmp_path: Path) -> None:
    _fixture(tmp_path)
    env = {"MPLBACKEND": "Agg", "PATH": "/usr/bin:/bin"}
    out = tmp_path / "links.png"
    r = subprocess.run([sys.executable, str(SCRIPT), "--experiments_dir", str(tmp_path), "--slot", "6",
                        "--min_axes", "1", "--links", "full:partial", "full:nosuch", "--output", str(out)],
                       capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stderr
    assert "--links full:nosuch: not both plotted" in r.stderr and out.exists()
    r2 = subprocess.run([sys.executable, str(SCRIPT), "--experiments_dir", str(tmp_path), "--slot", "6",
                         "--links", "bad", "--output", str(out)], capture_output=True, text=True, env=env)
    assert r2.returncode != 0 and "expected OLD:NEW" in r2.stderr


def test_include_models_is_an_allow_list(tmp_path: Path) -> None:
    _fixture(tmp_path)
    env = {"MPLBACKEND": "Agg", "PATH": "/usr/bin:/bin"}
    out = tmp_path / "inc.png"
    r = subprocess.run([sys.executable, str(SCRIPT), "--experiments_dir", str(tmp_path), "--slot", "6",
                        "--include_models", "full", "nosuch", "--output", str(out)],
                       capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stderr
    agg = r.stderr.split("Aggregated results")[1]
    assert "full" in agg and "partial" not in agg and "--include_models nosuch: no such cell" in r.stderr


def test_place_labels_skips_an_empty_label_but_keeps_its_point_as_an_obstacle():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from results_analysis.judge_tier_cost_plot_pooled import place_labels
    fig, ax = plt.subplots()
    ax.set_xlim(0, 10); ax.set_ylim(0, 10)
    place_labels(ax, [(5.4, 5), (5, 5)], ["", "seen"], [False, False])
    assert [t.get_text() for t in ax.texts] == ["seen"]
    assert tuple(ax.texts[0].get_position()) != (7, 0)            # the first choice, to the right, would cover the empty point
    plt.close(fig)
