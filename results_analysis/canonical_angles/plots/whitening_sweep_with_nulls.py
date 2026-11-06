"""Single-slot canonical-angles whitening sweep, r/t averaged, with two
random-null baselines.

What this plot shows
--------------------

For one chosen slot (default 3 = post-header newline), this wrapper plots
the goal-vs-nogoal canonical-angle spectrum under multiple whitening
regimes, averaged across kinds (``r`` and ``t``):

- ``raw``: identity (no whitening)
- ``soft_K=N`` for several values of N (default 1, 2, 4, 8, 16)
- ``lw``: Ledoit-Wolf shrinkage (``cov^{-1/2}`` whitening)

It also overlays two random-vector null baselines (faint grey):

- **uniform null** -- 30 vs 30 random ``N(0, I)`` vectors in hidden-dim
  space, averaged over ``--n_null_samples`` (default 25).  Sits near 90
  degrees throughout, since random subspaces in high-D are nearly
  orthogonal.
- **anisotropic null** -- 30 vs 30 random vectors drawn from
  ``N(0, Sigma)`` where ``Sigma`` has the held-out pool's actual
  eigenvalues for the top ``--null_aniso_frac`` (default 0.75) fraction
  of its rank, and a flat floor at the corresponding percentile
  eigenvalue elsewhere.  Captures "what CA would you see by chance,
  under the actual variance structure of the activation space?"

Subspaces and pool
------------------

Subspaces are ``combo_residual_theat_shifted`` marginals (origin=none):
one set of 30 goal-aligned per kind, one set of 30 non-goal-aligned per
kind.  Pass ``--aggregation combo_residual`` to disable the
theatricality shift.

Whitening pool defaults to ``roles+traits`` standalones with the 60
corresponding standalone vectors *held out* (a leave-one-out style guard
that the canonical-angles math depends on; using the combinations as the
pool is dishonest because every combination contributes to both
subspaces) plus the corpus ``default.pt`` as a single anchor row.
Pass ``--no-heldout`` to disable the leave-out, ``--no-augment`` to
drop the default.pt augmentation.

Caching
-------

Both the CA results and the null baselines are pickled to ``--cache_dir``
(default ``/tmp``).  The CA cache key includes ``(slot, layer, scope,
heldout, kinds, K)`` so re-running with new flags refits as needed but
re-runs with the same flags are instantaneous.  The slow part (~5 min on
the first run) is fitting Ledoit-Wolf on the 520x5120 pool; subsequent
re-renders take seconds.

Examples
--------

    # Default: slot 3, both kinds averaged, K=1..16, lw, two nulls
    uv run python -m results_analysis.canonical_angles.plots.whitening_sweep_with_nulls \\
        --output roger/canonical_angles_whitening_sweep_slot3.png

    # Slot 1 only, more null samples, custom K spectrum
    uv run python -m results_analysis.canonical_angles.plots.whitening_sweep_with_nulls \\
        --slot 1 --K 1 4 16 64 --n_null_samples 50 \\
        --output /tmp/sweep_slot1.png
"""

from __future__ import annotations

import argparse
import hashlib
import pickle
from pathlib import Path

import numpy as np

from assistant_axis import png_metadata, suptitle_with_specs
from assistant_axis.provenance import (
    InputSpec,
    current_data_subtree_input,
)
from .. import (
    AggSpec,
    CASpec,
    ComputeEnv,
    OriginSpec,
    WhiteningSpec,
    compute_ca_grid,
)
from ..core import _svd_basis, canonical_angles_deg
from ..data import (
    AUGMENTED_POOL_VERSION,
    DEFAULT_DATA_DIR,
    build_augmented_whitening_pool,
    build_subspace,
    build_whitening_pool,
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--data_dir", default=DEFAULT_DATA_DIR,
                   help=f"Cached vectors directory (default: {DEFAULT_DATA_DIR})")
    p.add_argument("--kinds", nargs="+", default=["r", "t"],
                   choices=["r", "t"],
                   help="Combination kinds to compute and average (default: r t)")
    p.add_argument("--slot", type=int, default=6,
                   help="Token-slot index (default: 6 = </think>; new judge-ρ "
                        "winner from May 2026 rejudge).  Pass --slot 3 (\\n) or "
                        "7 (\\n\\n post) to compare.  Slots 4-7 require the "
                        "8-slot Roger dataset; pass --data_dir accordingly.")
    p.add_argument("--layer", type=int, default=25,
                   help="Transformer layer (default: 25 -- Qwen-3-32B "
                        "optimum from rho_by_layer.py)")
    p.add_argument("--K", nargs="+", type=int, default=[1, 2, 3, 4, 6, 8],
                   help="K values for soft-K whitening (default: 1 2 4 8 16)")
    p.add_argument("--methods", nargs="+", default=["lw"],
                   choices=["lw", "oas"],
                   help="Additional whitening methods beyond raw and soft-K. "
                        "Default: ['lw'].  OAS is essentially identical to LW "
                        "on this data, so it's omitted by default.")
    p.add_argument("--scope", default="roles+traits",
                   choices=["roles", "traits", "roles+traits"],
                   help="Whitening pool scope (default: roles+traits)")
    p.add_argument("--no-heldout", dest="heldout", action="store_false",
                   help="Disable holding out subspace entities from the "
                        "whitening pool (default: held out -- 60 entities "
                        "removed for combo_residual subspaces).")
    p.set_defaults(heldout=True)
    p.add_argument("--no-augment", dest="augment", action="store_false",
                   help="Disable pool augmentation (default: augmented -- "
                        "pool gains the corpus default.pt as a single anchor "
                        "row).  Earlier versions also added mean(other-kind "
                        "combos) to give the whitener access to the "
                        "theatricality direction; that was nearly inert and "
                        "was dropped (see canonical_angles/README.md).")
    p.set_defaults(augment=True)
    p.add_argument("--n_null_samples", type=int, default=25,
                   help="Number of random samples for each null baseline "
                        "(default: 25)")
    p.add_argument("--null_aniso_frac", type=float, default=0.75,
                   help="Fraction of pool rank used as 'top' anisotropic "
                        "directions; eigenvalues beyond this fraction are "
                        "floored (default: 0.75)")
    p.add_argument("--null_seed", type=int, default=42,
                   help="RNG seed for null baseline sampling (default: 42)")
    p.add_argument("--cache_dir", default="/tmp",
                   help="Where to pickle CA + null caches (default: /tmp)")
    p.add_argument("--aggregation", default="combo_residual_theat_shifted",
                   choices=["combo_residual", "combo_residual_theat_shifted"],
                   help="Subspace aggregation (default: "
                        "combo_residual_theat_shifted -- residuals translated "
                        "to the additive origin via the per-(slot, layer) "
                        "theatricality shift; see canonical_angles/README.md "
                        "for the math). Pass 'combo_residual' to disable the "
                        "shift (e.g. for slot-0 analyses, or 'before' "
                        "comparisons).")
    p.add_argument("--output", required=True, help="Output PNG path")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Compute (with caching)
# ---------------------------------------------------------------------------

def _cache_key(prefix: str, *parts) -> str:
    """Stable filename slug for caching results."""
    s = repr(parts)
    h = hashlib.sha1(s.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}_{h}.pkl"


def compute_ca_results(args, env: ComputeEnv,
                       cache_path: Path) -> tuple[int, dict]:
    """Compute (or load from cache) the CA spectrum for each
    (kind, slot, whitening_label) combination at the chosen slot.

    Returns ``(pool_size, results_dict)``.
    """
    if cache_path.exists():
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    data_dir = Path(args.data_dir)
    pool_size = None
    results: dict = {}
    for kind in args.kinds:
        sub_a = tuple(build_subspace(data_dir, kind, "goal",
                                     args.aggregation, include_default=False))
        sub_b = tuple(build_subspace(data_dir, kind, "nogoal",
                                     args.aggregation, include_default=False))
        leave_out = (set(sub_a) | set(sub_b)) if args.heldout else set()
        if args.augment:
            pool = tuple(build_augmented_whitening_pool(
                data_dir, leave_out=leave_out, scope=args.scope))
        else:
            pool = tuple(build_whitening_pool(
                data_dir, scope=args.scope, leave_out=leave_out))
        pool_size = len(pool)
        print(f"kind={kind}: pool size = {pool_size} "
              f"(heldout = {args.heldout}, augment = {args.augment})")

        whitening_specs: list[tuple[str, WhiteningSpec]] = [
            ("raw", WhiteningSpec(method="raw")),
            *[(f"K={K}", WhiteningSpec(method="soft_K", K=K, pool=pool))
              for K in args.K],
            *[(m, WhiteningSpec(method=m, pool=pool)) for m in args.methods],
        ]
        for label, w_spec in whitening_specs:
            spec = CASpec(
                subspace_a=sub_a, subspace_b=sub_b,
                slot_indices=(args.slot,), slot_mode="single", layer=args.layer,
                aggregation=AggSpec(mode=args.aggregation, include_default=False),
                origin=OriginSpec(kind="none"),
                whitening=w_spec,
            )
            out = compute_ca_grid([spec], data_dir=data_dir, env=env)
            results[(kind, label)] = out[spec]

    payload = (pool_size, results)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(payload, f)
    print(f"Cached CA results to {cache_path}")
    return payload


def compute_null_baselines(args, env: ComputeEnv,
                           cache_path: Path) -> dict:
    """Compute (or load from cache) the two null baselines per kind.

    Returns ``{(kind, "uniform"|"aniso"): np.ndarray of length 30}``.
    """
    if cache_path.exists():
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    data_dir = Path(args.data_dir)
    rng = np.random.default_rng(args.null_seed)
    nulls: dict = {}

    for kind in args.kinds:
        sub_a = tuple(build_subspace(data_dir, kind, "goal",
                                     "combo_residual", include_default=False))
        sub_b = tuple(build_subspace(data_dir, kind, "nogoal",
                                     "combo_residual", include_default=False))
        leave_out = (set(sub_a) | set(sub_b)) if args.heldout else set()
        pool_entries = tuple(build_whitening_pool(
            data_dir, scope=args.scope, leave_out=leave_out))
        pool_mat = env.load_subspace_matrix(
            pool_entries, (args.slot,), "single", args.layer)
        n_vec = min(len(sub_a), len(sub_b))
        hidden = pool_mat.shape[1]

        # Uniform null
        nulls[(kind, "uniform")] = _null_uniform(
            rng, n_samples=args.n_null_samples, n_vec=n_vec, hidden=hidden)

        # Anisotropic null
        Vt_top, eig_top, floor = _fit_anisotropic(
            pool_mat, frac=args.null_aniso_frac)
        sqrt_eig_top = np.sqrt(eig_top)
        sqrt_floor = float(np.sqrt(floor))
        nulls[(kind, "aniso")] = _null_anisotropic(
            rng, Vt_top=Vt_top,
            sqrt_eig_top_minus_floor=sqrt_eig_top - sqrt_floor,
            sqrt_floor=sqrt_floor, hidden=hidden,
            n_samples=args.n_null_samples, n_vec=n_vec)
        print(f"kind={kind}: null fitted (pool top-{args.null_aniso_frac:.0%} "
              f"= {Vt_top.shape[0]} PCs, top eig = {eig_top[0]:.2f}, "
              f"floor = {floor:.4f})")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(nulls, f)
    print(f"Cached null baselines to {cache_path}")
    return nulls


def _null_uniform(rng, n_samples: int, n_vec: int, hidden: int) -> np.ndarray:
    """Average CA between two N(0, I) row-spans, averaged over n_samples."""
    samples = []
    for _ in range(n_samples):
        A = rng.standard_normal((n_vec, hidden))
        B = rng.standard_normal((n_vec, hidden))
        samples.append(canonical_angles_deg(_svd_basis(A), _svd_basis(B)))
    return np.mean(samples, axis=0)


def _null_anisotropic(rng, Vt_top: np.ndarray,
                      sqrt_eig_top_minus_floor: np.ndarray,
                      sqrt_floor: float, hidden: int,
                      n_samples: int, n_vec: int) -> np.ndarray:
    """Average CA between two N(0, Sigma) row-spans where Sigma has the
    top-r PCs from the pool weighted by their eigenvalues, and a flat
    floor on every other direction.

    Sample = sqrt(floor) * Z + (Z @ Vt_top.T) * (sqrt(eig_top) - sqrt(floor)) @ Vt_top
    """
    samples = []
    for _ in range(n_samples):
        Z_a = rng.standard_normal((n_vec, hidden))
        Z_b = rng.standard_normal((n_vec, hidden))
        proj_a = Z_a @ Vt_top.T
        proj_b = Z_b @ Vt_top.T
        A = sqrt_floor * Z_a + (proj_a * sqrt_eig_top_minus_floor) @ Vt_top
        B = sqrt_floor * Z_b + (proj_b * sqrt_eig_top_minus_floor) @ Vt_top
        samples.append(canonical_angles_deg(_svd_basis(A), _svd_basis(B)))
    return np.mean(samples, axis=0)


def _fit_anisotropic(pool_mat: np.ndarray, frac: float
                     ) -> tuple[np.ndarray, np.ndarray, float]:
    """Fit (Vt_top, variance_top, floor_variance) from a centered pool."""
    centered = pool_mat - pool_mat.mean(axis=0, keepdims=True)
    _U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    rank = int(np.sum(S > S[0] * 1e-6))
    n_top = max(int(frac * rank), 1)
    n_top = min(n_top, len(S) - 1)
    dof = max(centered.shape[0] - 1, 1)
    eig_top = (S[:n_top] ** 2) / dof
    floor = float((S[n_top] ** 2) / dof)
    return Vt[:n_top], eig_top, floor


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

LW_COLOR  = "#777777"
OAS_COLOR = "#8B4513"


def _build_K_palette(Ks: list[int]) -> dict:
    """Map each soft-K value to a plasma sample.

    Spacing is uniform in *list-position* (not in K itself), so the
    colors step through plasma in lock-step with the K values supplied
    on the CLI -- which gives finer color spacing automatically when
    the user passes a denser K set.  Plasma runs dark purple → magenta
    → orange → yellow; we stop at 0.9 to avoid the bright-yellow
    extreme that's hard to read on white.
    """
    import matplotlib.pyplot as plt
    Ks_sorted = sorted(set(Ks))
    if len(Ks_sorted) == 1:
        return {Ks_sorted[0]: plt.cm.plasma(0.5)}
    ts = np.linspace(0.0, 0.9, len(Ks_sorted))
    return {K: plt.cm.plasma(t) for K, t in zip(Ks_sorted, ts)}


def make_plot(args, pool_size: int,
              results: dict, nulls: dict,
              inputs: list[InputSpec] | None = None) -> None:
    """Render the slot-N plot, raw on top, K spectrum below, lw separately,
    nulls in faint grey at the back."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def avg(label: str) -> np.ndarray:
        return np.mean([results[(k, label)] for k in args.kinds], axis=0)

    def avg_null(kind_str: str) -> np.ndarray:
        return np.mean([nulls[(k, kind_str)] for k in args.kinds], axis=0)

    fig, ax = plt.subplots(1, 1, figsize=(11, 8))

    # Null baselines (drawn first, behind everything).
    u_null = avg_null("uniform")
    a_null = avg_null("aniso")
    x_null = np.arange(1, len(u_null) + 1)
    ax.plot(x_null, u_null, color="#bbbbbb", linestyle="--", linewidth=1.2,
            label="null: uniform random", alpha=0.85, zorder=1)
    ax.plot(x_null, a_null, color="#bbbbbb", linestyle=":",  linewidth=1.4,
            label=f"null: anisotropic ({int(args.null_aniso_frac*100)}% PCs)",
            alpha=0.85, zorder=1)

    # Mapping from internal compute keys to legend display names.
    # Internal labels in `results` are "K=N", "lw", "oas", "raw"; the
    # display names spell out the methods more fully for legend clarity.
    def _display(key: str) -> str:
        if key.startswith("K="):
            return f"{key} soft"
        if key == "lw":
            return "Ledoit-Wolf"
        if key == "oas":
            return "Oracle Approximating Shrinkage"
        return key

    # Order: lw at bottom of z, K=largest down to K=smallest, raw last
    # (on top).  Build a plasma palette over the actual K set so the
    # colors step uniformly through purple→magenta→orange→yellow no
    # matter how dense the K set is.
    palette = _build_K_palette(args.K)
    draw_order: list[tuple[str, str, object, float]] = []
    # (compute_key, display_name, color, linewidth)
    if "lw" in args.methods:
        draw_order.append(("lw", _display("lw"), LW_COLOR, 1.4))
    if "oas" in args.methods:
        draw_order.append(("oas", _display("oas"), OAS_COLOR, 2.6))
    for K in sorted(args.K, reverse=True):
        key = f"K={K}"
        draw_order.append((key, _display(key), palette[K], 1.6))
    draw_order.append(("raw", _display("raw"), "black", 1.6))

    for i, (compute_key, display_name, color, lw) in enumerate(draw_order):
        ang = avg(compute_key)
        x = np.arange(1, len(ang) + 1)
        ax.plot(x, ang, marker="o", markersize=4.5, linewidth=lw,
                color=color, label=display_name, alpha=0.95, zorder=10 + i)

    # Reorder legend: raw, K=1..K=N, lw/oas, then nulls.
    handles, labels = ax.get_legend_handles_labels()
    legend_order = (
        [_display("raw")]
        + [_display(f"K={K}") for K in sorted(args.K)]
        + [_display(m) for m in ("lw", "oas") if m in args.methods]
        + ["null: uniform random",
           f"null: anisotropic ({int(args.null_aniso_frac*100)}% PCs)"]
    )
    legend_idx = [labels.index(l) for l in legend_order if l in labels]
    ax.legend([handles[i] for i in legend_idx],
              [labels[i] for i in legend_idx],
              fontsize=10, loc="lower right", title="whitening regime")

    ax.axhline(y=90, color="gray", linestyle=":", alpha=0.4)
    ax.set_xlabel("Canonical Angle Index (sorted)", fontsize=12)
    ax.set_ylabel("Canonical Angle (degrees)", fontsize=12)
    ax.set_title(_slot_pretty(args.slot), fontsize=14)
    ax.set_ylim(0, 95)
    ax.set_xlim(0, 31)
    ax.grid(True, alpha=0.3)

    kinds_str = " and ".join(args.kinds)
    pool_descr = (f"{pool_size} entries: {args.scope}"
                  + (" (60 held out)" if args.heldout else " (no leave-out)")
                  + (" + default" if args.augment else ""))
    title_line = (f"Goal vs Non-goal canonical angles, slot {args.slot}, "
                  f"layer={args.layer}")
    spec_lines = [
        f"subspaces = {args.aggregation} (30 vs 30, {kinds_str} averaged), "
        f"origin=none",
        f"whitening pool = {pool_descr}",
    ]
    _, top_rect = suptitle_with_specs(fig, title_line, spec_lines)
    fig.tight_layout(rect=(0, 0, 1, top_rect))
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight",
                metadata=png_metadata(title=title_line, inputs=inputs))
    print(f"Wrote {out_path}")


def _slot_pretty(slot: int) -> str:
    pretty = {
        0: "Slot 0 (Body mean)",
        1: "Slot 1 (<|im_start|>)",
        2: "Slot 2 (assistant)",
        3: "Slot 3 (\\n)",
        4: "Slot 4 (<think>)",
        5: "Slot 5 (\\n\\n inside think)",
        6: "Slot 6 (</think>)",
        7: "Slot 7 (\\n\\n final)",
    }
    return pretty.get(slot, f"Slot {slot}")


def _build_inputs(data_dir: Path, args: argparse.Namespace) -> list[InputSpec]:
    """Provenance inputs for one run.

    Declares the goal/nogoal marginal subtrees actually used (per
    ``--kinds``); the theatricality axis subtree when shifted; and the
    pool subtrees feeding both the K-soft / lw / oas whitenings and
    the anisotropic-null fit (the uniform null is data-independent).
    """
    extras = {"slot": str(args.slot), "layer": str(args.layer),
              "kinds": ",".join(args.kinds),
              "K": ",".join(str(k) for k in args.K),
              "methods": ",".join(args.methods),
              "aggregation": args.aggregation,
              "scope": args.scope,
              "heldout": "1" if args.heldout else "0",
              "augment": "1" if args.augment else "0",
              "n_null_samples": str(args.n_null_samples),
              "null_aniso_frac": str(args.null_aniso_frac),
              "null_seed": str(args.null_seed)}
    inputs: list[InputSpec] = []
    needs_r = "r" in args.kinds
    needs_t = "t" in args.kinds
    if needs_r:
        inputs.append(current_data_subtree_input(
            data_dir, "combinations/vectors/derived/marginals/r_goal",
            dep_key="r_goal_marginals", extras=extras))
        inputs.append(current_data_subtree_input(
            data_dir, "combinations/vectors/derived/marginals/r_nogoal",
            dep_key="r_nogoal_marginals", extras=extras))
    if needs_t:
        inputs.append(current_data_subtree_input(
            data_dir, "combinations/vectors/derived/marginals/t_goal",
            dep_key="t_goal_marginals", extras=extras))
        inputs.append(current_data_subtree_input(
            data_dir, "combinations/vectors/derived/marginals/t_nogoal",
            dep_key="t_nogoal_marginals", extras=extras))
    if args.aggregation == "combo_residual_theat_shifted":
        inputs.append(current_data_subtree_input(
            data_dir, "combinations/vectors/derived/axis",
            dep_key="theatricality_axis", extras=extras))
    if "roles" in args.scope:
        inputs.append(current_data_subtree_input(
            data_dir, "roles/vectors", dep_key="roles_vectors",
            extras=extras))
    if "traits" in args.scope:
        inputs.append(current_data_subtree_input(
            data_dir, "traits/vectors", dep_key="traits_vectors",
            extras=extras))
    return inputs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()
    cache_dir = Path(args.cache_dir)
    cache_key_args = (
        args.data_dir, tuple(sorted(args.kinds)), args.slot, args.layer,
        tuple(sorted(args.K)), tuple(sorted(args.methods)),
        args.scope, args.heldout, args.augment, args.aggregation,
        AUGMENTED_POOL_VERSION if args.augment else "no-aug",
    )
    ca_cache = cache_dir / _cache_key("ca_sweep", *cache_key_args)
    null_cache = cache_dir / _cache_key(
        "ca_nulls", *cache_key_args, args.n_null_samples,
        args.null_aniso_frac, args.null_seed,
    )

    env = ComputeEnv(data_dir=Path(args.data_dir))
    print(f"CA cache:    {ca_cache}")
    print(f"Null cache:  {null_cache}\n")

    pool_size, results = compute_ca_results(args, env, ca_cache)
    nulls = compute_null_baselines(args, env, null_cache)

    inputs = _build_inputs(Path(args.data_dir), args)
    make_plot(args, pool_size=pool_size, results=results, nulls=nulls,
              inputs=inputs)

    # Compact summary
    print(f"\n--- Summary (slot {args.slot}, kinds averaged) ---")
    print(f"{'regime':12s}  {'first':>7s}  {'med':>7s}  {'last':>7s}")
    labels = (
        ["raw"]
        + [f"K={K}" for K in sorted(args.K)]
        + list(args.methods)
        + [f"null:{n}" for n in ("uniform", "aniso")]
    )
    for label in labels:
        if label.startswith("null:"):
            kind_str = label.split(":", 1)[1]
            ang = np.mean([nulls[(k, kind_str)] for k in args.kinds], axis=0)
        else:
            ang = np.mean([results[(k, label)] for k in args.kinds], axis=0)
        print(f"  {label:10s}  "
              f"{float(np.nanmin(ang)):7.2f}  "
              f"{float(np.nanmedian(ang)):7.2f}  "
              f"{float(np.nanmax(ang)):7.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
