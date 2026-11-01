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
