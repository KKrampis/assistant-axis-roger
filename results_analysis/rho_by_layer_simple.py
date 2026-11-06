#!/usr/bin/env python3
"""Single-curve "correlation vs layer" plot — a simplified consumer
view of :mod:`results_analysis.rho_by_layer`'s 8-panel L-sweep output.

Reads the cached ``rho_by_layer_L.json`` (or ``rho_by_layer_K.json``)
produced by ``rho_by_layer.py`` and renders ONE curve:

* one slot (default: 6, the canonical ``</think>`` token)
* one whitening level (default: 0 = raw, no whitening)
* the canonical response × desc+inst blend
  (``DEFAULT_RESPONSE_DI_WEIGHT * response_rho +
  (1 - DEFAULT_RESPONSE_DI_WEIGHT) * di_rho``) applied per layer

Useful when you want the headline "where is the per-layer ρ peak
under the canonical blend?" view without the layout overhead of the
full 8-panel rho_by_layer plot.  No SVD; just reads the JSON cache
and emits a PNG with proper ``png_metadata`` provenance.

CLI::

    # Default: slot 6 / L=0 / canonical blend, write to
    # rho_by_layer_simple_slot6_raw.png next to the input JSON.
    uv run python results_analysis/rho_by_layer_simple.py

    # Different slot or whitening level
    uv run python results_analysis/rho_by_layer_simple.py --slot 3 --L 2

    # Different blend weight (e.g. inspect at 0.5 response weight)
    uv run python results_analysis/rho_by_layer_simple.py --response_di_weight 0.5
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from assistant_axis import json_metadata, png_metadata  # noqa: E402
from assistant_axis.judge_score_combine import (   # noqa: E402
    DEFAULT_RESPONSE_DI_WEIGHT,
)
from assistant_axis.provenance import (   # noqa: E402
    InputSpec, current_file_input,
)


DEFAULT_INPUT = REPO_ROOT / "roger/axis_judge_experiments/rho_by_layer_L.json"


def _filter_to_slot_and_L(rows: np.ndarray, slot: int, L: int) -> np.ndarray:
    """Filter ``rows`` (cols: slot, layer, K, rho) to one (slot, L).
    Returns rows sorted by layer."""
    mask = (rows[:, 0] == slot) & (rows[:, 2] == L)
    sub = rows[mask]
    return sub[np.argsort(sub[:, 1])]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input", type=Path, default=DEFAULT_INPUT,
        help="Path to rho_by_layer_L.json (or _K.json) from rho_by_layer.py.",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output PNG path (default: rho_by_layer_simple_slot{S}_"
             "raw.png next to --input).",
    )
    parser.add_argument(
        "--slot", type=int, default=6,
        help="Token slot to plot (default: 6 = </think>, the canonical "
             "operating slot for response-mode steering).",
    )
    parser.add_argument(
        "--L", type=int, default=0,
        help="Whitening level (default: 0 = raw / no whitening). For "
             "soft_shear sweep, L=1..6 are progressively softer-shear; "
             "for soft-K sweep, K=1..6 are progressively softer-whitening.",
    )
    parser.add_argument(
        "--response_di_weight", type=float,
        default=DEFAULT_RESPONSE_DI_WEIGHT,
        help=f"Weight on the response cohort in the final blend "
             f"(default: {DEFAULT_RESPONSE_DI_WEIGHT} = "
             f"DEFAULT_RESPONSE_DI_WEIGHT).  1 - this goes on desc+inst.",
    )
    args = parser.parse_args()

    inputs: list[InputSpec] = [
        current_file_input(
            dep_key="producer_script",
            path=Path(__file__),
            extras={
                "slot": str(args.slot),
                "L": str(args.L),
                "response_di_weight": str(args.response_di_weight),
            },
        ),
        current_file_input(
            dep_key="rho_by_layer_json",
            path=args.input,
            extras={"slot": str(args.slot), "L": str(args.L)},
        ),
    ]

    raw = json.loads(args.input.read_text())["result"]
    di_rows = np.array(raw["desc_inst"])     # cols: (slot, layer, K, rho)
    rs_rows = np.array(raw["responses"])

    di_sub = _filter_to_slot_and_L(di_rows, args.slot, args.L)
    rs_sub = _filter_to_slot_and_L(rs_rows, args.slot, args.L)
    if len(di_sub) != len(rs_sub):
        print(f"WARN: di has {len(di_sub)} layers but responses has "
              f"{len(rs_sub)}; truncating to overlap")
    n_layers = min(len(di_sub), len(rs_sub))
    layers = di_sub[:n_layers, 1].astype(int)
    rho_di = di_sub[:n_layers, 3]
    rho_rs = rs_sub[:n_layers, 3]

    w = float(args.response_di_weight)
    rho_combined = w * rho_rs + (1.0 - w) * rho_di

    n_axes_di = raw["n_axes_di"]
    n_axes_resp = raw["n_axes_resp"]

    peak_layer = int(layers[np.argmax(rho_combined)])
    peak_rho = float(rho_combined.max())
    print(f"Primary peak combined rho = {peak_rho:.4f} at layer {peak_layer}")

    # Secondary peak in layer-40..60 region
    sec_window = (layers >= 40) & (layers <= 60)
    if sec_window.any():
        sec_idx = np.where(sec_window)[0][np.argmax(rho_combined[sec_window])]
        sec_layer = int(layers[sec_idx])
        sec_rho = float(rho_combined[sec_idx])
        print(f"Secondary peak = {sec_rho:.4f} at layer {sec_layer}")
    else:
        sec_layer = sec_rho = None

    # ---- plot ----
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(layers, rho_combined, color="black", lw=2.2, zorder=5,
            label="correlation")

    ax.axvline(peak_layer, color="grey", lw=0.8, linestyle="--", alpha=0.6)
    ax.annotate(f"primary peak\nlayer {peak_layer}\nρ={peak_rho:.3f}",
                xy=(peak_layer, peak_rho),
                xytext=(peak_layer + 3, peak_rho + 0.02),
                fontsize=10,
                arrowprops=dict(arrowstyle="->", lw=0.6, color="grey"))
    if sec_layer is not None:
        ax.axvline(sec_layer, color="grey", lw=0.8, linestyle="--", alpha=0.6)
        ax.annotate(f"secondary peak\nlayer {sec_layer}\nρ={sec_rho:.3f}",
                    xy=(sec_layer, sec_rho),
                    xytext=(sec_layer - 8, sec_rho + 0.04),
                    fontsize=10,
                    arrowprops=dict(arrowstyle="->", lw=0.6, color="grey"))

    # Slot label hint -- canonical slots in this project are
    # 0 = body mean, 3 = '\n', 6 = '</think>', 7 = '\n\n post'.
    slot_labels = {
        0: "body mean",
        3: r"$\langle\backslash\mathrm{n}\rangle$",
        6: r"$\langle/\mathrm{think}\rangle$",
        7: r"$\langle\backslash\mathrm{n}\backslash\mathrm{n}\ \mathrm{post}\rangle$",
    }
    slot_caption = slot_labels.get(args.slot, f"slot {args.slot}")
    whitening_caption = "raw (no whitening)" if args.L == 0 else f"L={args.L}"
    blend_caption = (f"{w:.2f} response + {1-w:.2f} desc+inst"
                     f"   (response={n_axes_resp} axes, desc+inst={n_axes_di} axes)")

    title_line = "Correlation between activations and judging vs layer"
    subtitle = (f"header token {args.slot} ({slot_caption}), "
                f"{whitening_caption}, canonical response/desc+inst mix "
                f"({blend_caption})")

    ax.set_xlabel("Transformer layer", fontsize=12)
    ax.set_ylabel("Mean per-axis Spearman ρ", fontsize=12)
    ax.set_title(f"{title_line}\n{subtitle}", fontsize=12, pad=10)
    ax.set_xlim(0, max(63, layers.max()))
    ax.set_ylim(0, max(0.85, rho_combined.max() * 1.06))

    ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.xaxis.set_minor_locator(MultipleLocator(2))
    ax.grid(which="major", axis="both", alpha=0.30)
    ax.grid(which="minor", axis="x", alpha=0.12, linewidth=0.6)

    ax.legend(loc="lower right", fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    if args.output is None:
        suffix = "raw" if args.L == 0 else f"L{args.L}"
        out = args.input.parent / f"rho_by_layer_simple_slot{args.slot}_{suffix}.png"
    else:
        out = args.output

    plt.savefig(out, dpi=150, bbox_inches="tight",
                metadata=png_metadata(title=title_line, inputs=inputs))
    plt.close(fig)
    print(f"Wrote {out}")

    # Also dump the per-layer numbers as a JSON sidecar so downstream
    # consumers can read the combined curve without re-running this.
    sidecar = out.with_suffix(".json")
    sidecar.write_text(json.dumps(json_metadata(
        {
            "slot": args.slot,
            "L": args.L,
            "response_di_weight": w,
            "layers": layers.tolist(),
            "rho_combined": rho_combined.tolist(),
            "rho_response_only": rho_rs.tolist(),
            "rho_desc_inst_only": rho_di.tolist(),
            "primary_peak": {"layer": peak_layer, "rho": peak_rho},
            "secondary_peak": (
                {"layer": sec_layer, "rho": sec_rho}
                if sec_layer is not None else None
            ),
            "n_axes_di": n_axes_di,
            "n_axes_resp": n_axes_resp,
        },
        title="Correlation between activations and judging vs layer",
        inputs=inputs,
    ), indent=2))
    print(f"Wrote {sidecar}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
