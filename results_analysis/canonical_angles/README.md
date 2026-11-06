# Canonical Angles

A configurable canonical-angles analysis tool with separated computation, plot
helpers, and per-plot wrappers.

Canonical angles (a.k.a. principal angles) measure how aligned two linear
subspaces are: the smallest angle is 0 if they share a direction, and 90 if
they are completely orthogonal. We use them here to ask things like *"how
orthogonal are the goal and non-goal subspaces of activation space?"* and
*"how does PCA whitening affect that orthogonality?"*.

## Layout

```
results_analysis/canonical_angles/
  __init__.py          Public API surface (re-exports core types/functions)
  core.py              CASpec, AggSpec, OriginSpec, WhiteningSpec
                       compute_ca / compute_ca_grid (with caching)
                       canonical_angles_deg primitive
  data.py              Entity loading, subspace builders, whitening pool,
                       slot composition
  whitening.py         WhiteningBasis: raw / soft_K / lw / oas
                       fit_whitening, parse_whitening_spec
  plot_helpers.py      plot_per_slot_panels, plot_overlay_curves, plot_grid
  ca1_plane.py         CA1Decomposition (a_1 / b_1 / e_+ / e_- / shear),
                       compute_ca1_decomposition, plot_ca1_plane_pre_post_shear,
                       collision-aware label placement
  README.md            (this file)
  plots/
    goal_vs_nogoal_mutual.py            (1st canonical wrapper)
    combos_vs_traits_roles_pooled.py    (2nd canonical wrapper)
    ca1_plane_pre_post_shear.py         CA1-plane visualization, pre/post shear
```

## What the compute layer does

A single computation is described by a `CASpec` carrying:

| field | meaning |
|---|---|
| `subspace_a` | tuple of `(etype, name)` entries -- one canonical subspace |
| `subspace_b` | tuple of `(etype, name)` entries -- the other |
| `slot_indices`, `slot_mode` | which slot(s), and how to combine them (`single` / `avg` / `concat`) |
| `layer` | transformer layer index |
| `aggregation` (`AggSpec`) | how to construct the subspace from raw data: `standalone` / `combo_centroid` / `combo_residual` |
| `origin` (`OriginSpec`) | what to subtract before computing the SVD basis: `none` / `default` / `self_mean` / `pool_mean` / `full_combo_mean` / `paired` |
| `whitening` (`WhiteningSpec`) | optional PCA whitening: `raw` / `soft_K` / `lw` / `oas`, with a `pool` (tuple of entries) used for the basis fit |

`compute_ca(spec, env)` returns one numpy array of canonical angles in
degrees, sorted ascending. `compute_ca_grid(specs, data_dir)` takes a list of
specs and returns a `dict[CASpec, np.ndarray]`, sharing cached intermediates
(loaded vectors, fitted whiteners, computed origins) between specs that share
those intermediates.

## The 7 configuration variables

The wrappers ultimately expose seven knobs that drive the whole tool:

| # | variable | values | typical default |
|---|---|---|---|
| 1 | slots | int / list / `'all'`; each entry can be `'avg(i,j)'` or `'concat(i,j)'` to merge slots | last header slot |
| 2 | kind | `r` / `t` / `both` (parallel) / `combined` (60 vs 60 union) | `both` |
| 3 | layer | int / list / `'all'` | `24` |
| 4 | whitening | `raw` / `soft_K=N` / `lw` / `oas`; or a list | `raw` for the goal-vs-nogoal wrapper, sweep of K values for the combos wrapper |
| 5 | whitening pool | scope (`roles` / `traits` / `roles+traits` / `combinations`) and `leave_out` toggle, plus an `augment` toggle that adds `default + mean(other-kind combos)` (recommended for combo_residual analyses) | `roles+traits, leave_out=true, augment=true` |
| 6 | aggregation | `standalone` / `combo_residual` / `combo_residual_theat_shifted` (`combo_centroid` deprecated) | `combo_residual_theat_shifted` (residual + theatricality shift to the additive origin; see "Theatricality shift" below). Pass `--aggregation combo_residual` to disable the shift (recommended for slot 0). |
| 7 | origin | `none` / `default` / `self_mean` / `pool_mean` / `full_combo_mean` / `paired` | `none` (residuals already have a semantic zero) |

Aggregation modes:

- **`standalone`**: subspace built from standalone trait or role vectors
  (the directly-prompted activations stored under `traits/vectors/` and
  `roles/vectors/`).
- **`combo_residual`**: per-axis residual marginals read from on-disk
  `r_goal/`, `r_nogoal/`, `t_goal/`, `t_nogoal/`. Each entry is the mean
  over partners B of `combo(A, B) - standalone(B)` -- i.e., A's net
  effect with the partner's main effect subtracted. This is the only
  honest reduction under partner-set imbalance (different roles paired
  with different trait subsets), which our 898/900 grid already exhibits
  (and which the new clean trait pairs will exhibit much more prominently).
  Produced by
  [`results_analysis/compute_combo_marginals.py`](../compute_combo_marginals.py).
  Use this aggregation directly only when you specifically want the
  unshifted residual (e.g. slot-0 analyses); see the slot-0 caveat under
  "Theatricality shift".
- **`combo_residual_theat_shifted`** *(default)*: same on-disk file as
  `combo_residual`, but the env adds the per-(slot, layer) theatricality
  shift from `combinations/vectors/theatricality_axis.pt` on the fly.
  This relocates each residual to the additive model's true origin
  (where `combo ≈ role + trait` directly, with no constant offset
  along the "performative-persona" axis `v_theat` shared by all four
  `combo_{r,t}_{goal,nogoal}` subspaces).  On header slots (1, 2, 3) this
  restores monotonically-increasing CA1 with K under soft whitening; on
  slot 0 (body mean) it makes things slightly worse, indicating the
  theatricality offset is a header-slot phenomenon.  See the "Theatricality
  shift" section below for the math and rationale.
- **`combo_centroid`** *(deprecated)*: the unbiased-grid version that the
  historical plots used. No longer exposed via the wrapper CLIs.  Biased
  on incomplete grids; the residual aggregation is a strict
  generalization (identical on a complete grid, more honest otherwise).
  Reads from the legacy snapshot dirs `r_goal_legacy_centroid/` etc.
  (preserved on disk for bit-exact reproduction); falls back to
  on-the-fly recomputation if the snapshot is absent.  Kept only for
  programmatic access via :func:`combo_centroid_subspace`.

On a complete grid, residual and centroid produce identical subspaces (up
to a global shift that `self_mean` / `full_combo_mean` origins absorb).
On a sparse or biased grid, the centroid subspace's reported angles
artificially shrink because shared partner-trait bias inflates the
similarity between roles; the residual subspace stays close to the true
full-grid spectrum.

Origin modes:

- **`none`** *(default for residual aggregation)*: no centering. The
  natural choice for residual marginals because they are already deltas
  with a semantic zero (`residual[A] = 0` literally means "A adds nothing
  on top of its partner"). Self-centering would discard the radial
  "average effect" direction. Yields a 30-d subspace from 30 entities.
- **`self_mean`**: each subspace centered on its own centroid (different
  origins for A and B). Standard CA convention; numerically robust. With
  residual aggregation it removes one dimension (the average-effect
  direction) so the subspace is 29-d. Use when you want to compare only
  the *spread* of residuals, not their average direction.
- **`full_combo_mean`**: single shared origin = mean of all combinations of
  the relevant kind plus default. Used by the historical
  `canonical_angles_goal_vs_nogoal_mutual.png`; only meaningful when both
  subspaces live in raw activation space (i.e. for centroid aggregation).
- **`default`**: subtract the default activation vector.
- **`pool_mean`**: mean of the relevant entity pool (roles+traits etc.).
- **`paired`**: per-vector partner offset, valid with `combo_residual`.

## Whitening pool

Default pool: held-out standalone roles + traits + corpus
``combinations/default.pt``.  Pass `--no-augment` to drop the default
and use just the standalones.  ``build_augmented_whitening_pool`` in
`canonical_angles/data.py` is the helper.

### Why we *don't* inject the theatricality direction into the pool

There's a tempting symmetric alternative to the theatricality shift
documented below: instead of moving the *subspace* to the additive
origin, move the *whitener* by giving the pool access to the
theatricality direction ``v_theat`` so soft-K can shrink it directly.
We tried that and it didn't work, for instructive reasons.

**Attempt 1 -- add `mean_<other-kind>_combos` to the pool.** The
centroid of the OTHER kind's combinations carries the same
theatricality offset as the analysed kind (both kinds share the
additive structure), so adding it as a single held-out row should
inject ``v_theat`` into the pool's PCA without leaking same-kind
information.  Empirically, **nearly inert**:

- Variance along ``v_theat`` in the default-augmented pool (slot 3,
  layer 25): 33.5, vs 32.2 in the un-augmented standalone-only pool --
  a 4% increase from one new row out of 520.
- Top-100 PCs cover 78.5% of ``v_theat`` (aug) vs 70.8% (un-aug) --
  modest improvement only in the long tail.
- CA1 difference at K=4..16 between
  ``combo_residual_theat_shifted + augmented`` and
  ``combo_residual_theat_shifted + default-only``: ≤0.01°. Only LW
  CA1 noticeably differs (a few degrees), and we don't use LW for the
  primary analysis.
- See [`roger/canonical_angles_theat_shift_3panel_4slot.png`](../../roger/canonical_angles_theat_shift_3panel_4slot.png)
  and [`canonical_angles_theat_shift_3panel_slot3.png`](../../roger/canonical_angles_theat_shift_3panel_slot3.png)
  for the head-to-head.

**Attempt 2 -- 60 copies of `mean_t_combos`.** Forcing the issue by
inserting 60 identical copies of the cross-kind centroid did move CA1
substantially at moderate K, but heavily distorted the rest of the PC
structure (the duplicated row dominated multiple top PCs and pulled
them toward a single axis).  The "fix" was an artifact: it made K=4
soft whitening behave as a near-projection onto the duplicate's
direction.  Not a meaningful intervention.

**Why the standalone pool already covers ``v_theat`` adequately.**
Individual standalone roles and traits each project +11 to +14 onto
``v_theat`` on average, so the pool's natural variance has a real
component along that direction; it's just spread across PCs rather
than concentrated.  Including ``default`` adds a single
strongly-negative-projection anchor (-7) that pairs with the average
positive projection to give the whitener a more identifiable axis,
without distorting the PC structure the way 60 copies of a single
direction does.

**The principled fix is on the subspace side.** The
``combo_residual_theat_shifted`` aggregation cancels the theatricality
offset *deterministically* via a single per-(slot, layer) translation
derived from ``default - μ_pool``, removing the shared common-mode
component without any pool gymnastics.  Doing it on the subspace side
is also conceptually cleaner: the offset is a property of the residual
construction (the constant ``X · v_theat`` term in the additive model),
not of the whitening regime, so it's correctly fixed where it arises.

The unused `mean_r_combos.pt` / `mean_t_combos.pt` artifacts continue
to be produced by ``compute_combo_marginals.py`` for ad-hoc analyses
(they're cheap to compute), but the canonical-angles tool no longer
reads them.

## Theatricality shift (`combo_residual_theat_shifted`) — the default

This is the **default aggregation** for all wrappers.  The plain
`combo_residual` is left as an opt-in for "before" comparisons and for
slot-0 / non-header analyses (see caveats below).

### Why a shift is needed: the additive model of role+trait vectors

A combination's activation vector decomposes empirically into

    combo(role, trait) ≈ role + trait − μ_pool + X · v_theat

where `μ_pool` is the mean of the held-out roles+traits standalones and
`v_theat` is a single, slot/layer-specific unit direction (see
`compute_theatricality_axis` in
[`compute_combo_marginals.py`](../compute_combo_marginals.py)).  The
last term is the only systematic **non-additive** component: a constant
offset along one axis that survives partial-pooling of any combo, role,
or trait subset (see
[`roger/variance_decomp_bars.png`](../../roger/variance_decomp_bars.png)
for the original April-23 variance decomposition).

Semantically `v_theat` reads as a "performative-persona" direction —
the mean residual of "be a character + a personality trait" relative
to the pure additive prediction.  X (the scalar coordinate along
`v_theat`) is roughly constant across combinations of a given kind at
a given (slot, layer), confirming it's an **offset**, not a per-combo
attribute.

Now consider the `combo_residual` marginal of role A:

    r_goal[A] = E_B [ combo(A, B) − trait[B] ]
              ≈ A + (E_B trait[B] − μ_pool) + X · v_theat
              ≈ A + bias_trait + X · v_theat

— role A plus a partner-mean bias plus the constant theatricality
offset.  The four residual subspaces (`r_goal`, `r_nogoal`, `t_goal`,
`t_nogoal`) all carry approximately the **same** `X · v_theat` term, so
their affine spans share that direction.  Geometrically: instead of
sitting around their natural additive origin (`bias_trait` for r_goal,
analogously for the others), the residuals are uniformly translated by
`X · v_theat`.

That uniform translation has two visible consequences:

1. **Whitening fails to shrink theatricality.** Soft-K whitening with
   the standalone-only roles+traits pool cannot see `v_theat` (it isn't
   in the pool's PCs), so it cannot shrink it.  As K grows, the
   subspaces collapse toward `v_theat`, and CA1 between `r_goal` and
   `r_nogoal` first rises (genuine common variance shrinking) then
   *falls* (residual collapse along `v_theat`).
2. **The "no-centering" choice (`origin=none`) is misaligned.** With
   plain `combo_residual`, "no centering" means "centered on
   `bias_trait + X · v_theat`", not on the additive origin.  We want
   the latter.

### What the shift does

The theatricality axis file
(`combinations/vectors/theatricality_axis.pt`) stores per-(slot, layer)
the unit vector `v_theat` and the scalar

    default_offset = (default − μ_pool) · v_theat

i.e. the theatricality coordinate of the corpus default vector relative
to the pool mean.  The shift applied by `combo_residual_theat_shifted`
is

    shift = default_offset · v_theat

(the *vector* form of the scalar offset).  Adding this shift to every
entry of every `combo_residual` marginal:

- **Translates the subspaces back to the additive origin.**  After
  shifting, `combo ≈ role + trait` directly, with no constant offset —
  the four residual subspaces are now positioned where pure vector
  arithmetic predicts.
- **Lets the augmented-pool whitener actually do its job.** With
  `v_theat` reachable in the pool's top PCs (via the `mean_t_combos`
  augmentation, see "Whitening pool augmentation" above), soft-K
  whitening can now shrink the shared theatricality variance, and CA1
  rises monotonically with K.

The choice of `default - μ_pool` as the reference is principled: the
default activation is the corpus baseline, and we want the additive
origin to coincide with "no role, no trait, ordinary helpful Assistant",
which is exactly the default's projection onto the theatricality axis.

### Empirical effect

Slot 3, layer 25, r and t averaged, K=8 soft whitening (augmented
pool):

| metric | combo_residual | combo_residual_theat_shifted |
|---|---:|---:|
| CA1 raw | 31° | 30° |
| CA1 K=4 | 37° | 36° |
| CA1 K=16 | 36° | 49° |
| CA1 K=64 | 28° | 61° |
| CA1 monotone in K | no — peaks at K=4 then drops | yes |

See [`roger/canonical_angles_theat_shift_4slot.png`](../../roger/canonical_angles_theat_shift_4slot.png)
and [`roger/canonical_angles_theat_shift_slot3.png`](../../roger/canonical_angles_theat_shift_slot3.png)
for the full per-slot before/after comparison.

### Slot-0 caveat (when not to shift)

On slot 0 (body-mean activation) the shift goes the **other** way —
CA1 *decreases* under the shift.  This is consistent with the
theatricality offset being a header-token phenomenon (the
"performative-persona" signal is encoded primarily in the chat-header
slots 1–3 where role/trait identity is established, not in the
post-response body mean).  For slot-0 analyses, pass
`--aggregation combo_residual` to disable the shift.

### Implementation

The shift is applied **lazily** in `ComputeEnv.get_entity` whenever an
entity's etype ends in `_theat_shifted`, so:

- The on-disk `r_goal/`, `r_nogoal/`, `t_goal/`, `t_nogoal/` marginals
  are unchanged — switching aggregation costs nothing on disk.
- `theatricality_axis.pt` is loaded once per env and cached.
- Subspace identity is preserved across kinds (the same shift is added
  to every kind's marginals at a given (slot, layer)).

To regenerate the axis file after changing the held-out pool, the
combination grid, or `compute_combo_marginals.py` itself:

    uv run python results_analysis/compute_combo_marginals.py

## Whitening regimes (`whitening.py`)

- `raw` -- identity transform
- `soft_K=N` -- rescale the top-N right-singular components of the
  centered pool to have effective sigma equal to sigma_{N+1}; lower
  components untouched. Equivalent to "soft" PCA whitening.
- `lw` -- Ledoit-Wolf shrinkage covariance, full `cov^{-1/2}` matrix
- `oas` -- Oracle Approximating Shrinkage, same structure as `lw`

CLI parsing in wrappers uses `parse_whitening_spec(s)`.

## Plot helpers

Three composable helpers in `plot_helpers.py`. Each takes a list of
"long-format" record dicts shaped like:

    {'panel': 'something', 'group': 'something', 'angles': np.ndarray, ...}

- `plot_per_slot_panels(records, panel_key, group_key, as_bars=...)` -- one
  panel per `panel_key` value, one curve (or set of bars) per `group_key`.
- `plot_overlay_curves(records, group_key)` -- single panel, multiple curves.
- `plot_grid(records, row_key, col_key, group_key=...)` -- N x M facet grid.

Wrappers compose these as needed.

## Existing wrappers

### `plots/goal_vs_nogoal_mutual.py`

Compares the goal subspace to the non-goal subspace of `r_*__*` (or
`t_*__*`) combinations. Default: `combo_residual` aggregation with
`origin=none` centering and no default vector (residuals already live in
a delta-space with a semantic zero, so no further centering is needed).

```bash
# Default (residual + origin=none), both kinds in parallel
uv run python -m results_analysis.canonical_angles.plots.goal_vs_nogoal_mutual \
    --output roger/canonical_angles_goal_vs_nogoal_mutual.png

# Whitening sweep, kind=combined (60 vs 60)
uv run python -m results_analysis.canonical_angles.plots.goal_vs_nogoal_mutual \
    --kinds combined --whitening raw soft_K=4 soft_K=128 \
    --slots 3 --output /tmp/test_sweep.png
```

### `plots/combos_vs_traits_roles_pooled.py`

Goal-vs-nogoal canonical angles swept across multiple soft-K whitening
regimes, with the whitening pool being **all combinations of the kind**.
The subspace pair is the same as `goal_vs_nogoal_mutual` -- this wrapper
shows that the same core handles different whitening configurations
cleanly. Defaults: `combo_residual` aggregation, `include_default=False`,
`origin=none`.

```bash
# Reproduce r-side
uv run python -m results_analysis.canonical_angles.plots.combos_vs_traits_roles_pooled \
    --kind r --output /tmp/test_combos_r.png

# Reproduce t-side
uv run python -m results_analysis.canonical_angles.plots.combos_vs_traits_roles_pooled \
    --kind t --output /tmp/test_combos_t.png

# Custom K sweep
uv run python -m results_analysis.canonical_angles.plots.combos_vs_traits_roles_pooled \
    --kind r --K 4 8 16 32 64 128 --output /tmp/k_sweep.png
```

### `plots/layer_sweep.py`

Goal-vs-nogoal canonical-angle spectrum across multiple transformer
layers, with `raw` and `K=3 soft` whitening shown side by side.
Defaults to a 17-layer set spanning `17, 21, 25, 27, 29, 31, ..., 53`
(centered on Qwen-3-32B's analysis layer 25 with 2-layer spacing on
the high side), slots `[0, 3]` (one row per slot), 2x2 panel grid,
rainbow-colored layer curves with early layers drawn on top.  Useful
for "is L25 still the right choice?" sanity checks: at slot 3 raw,
alignment between goal and non-goal residuals is roughly monotonic in
depth across this range.

```bash
# Default
uv run python -m results_analysis.canonical_angles.plots.layer_sweep \
    --output roger/canonical_angles_layer_sweep.png

# Different layer set, only slot 3
uv run python -m results_analysis.canonical_angles.plots.layer_sweep \
    --layers 12 24 36 48 60 --slots 3 \
    --output /tmp/layer_sweep_slot3_only.png
```

### `plots/whitening_sweep_with_nulls.py`

Single-slot whitening sweep with two random-vector null baselines
(uniform and pool-anisotropic).  Designed for slide decks: one big panel,
r and t kinds averaged, K spectrum colored purple -> teal -> yellow ->
orange -> red.  Pool defaults to held-out `roles+traits` standalones
(519 entries with the 60 corresponding standalones removed) plus the
corpus `default.pt` (520 entries total) -- the principled choice for
`combo_residual_theat_shifted` subspaces, since you genuinely cannot
hold out the all-combinations pool without emptying it.

The two null baselines (averaged over `--n_null_samples`, default 25,
cached) answer the question "is the observed CA spectrum more aligned
than chance?":

- **uniform null**: 30 vs 30 random `N(0, I)` vectors (sits at ~85 deg in
  high-D, the trivial upper bound).
- **anisotropic null**: 30 vs 30 random vectors from `N(0, Sigma)` with
  Sigma matching the held-out pool's eigenvalues for the top
  `--null_aniso_frac` (default 75%) of its rank, and a flat floor at the
  corresponding percentile elsewhere.  Captures "what does the activation
  space's anisotropy alone predict?"

Both CA results and null baselines are pickled to `--cache_dir`
(default `/tmp`) keyed on the relevant args; the slow part (LW fit on
the 520x5120 pool) takes ~80 s on the first call but subsequent
re-renders are seconds.

```bash
# Default: slot 3, both kinds averaged, K=1..16, lw, two nulls
uv run python -m results_analysis.canonical_angles.plots.whitening_sweep_with_nulls \
    --output roger/canonical_angles_whitening_sweep_slot3.png

# Different slot, custom K spectrum, more null samples
uv run python -m results_analysis.canonical_angles.plots.whitening_sweep_with_nulls \
    --slot 1 --K 1 4 16 64 --n_null_samples 50 \
    --output /tmp/sweep_slot1.png
```

## CA1-plane visualization (`plots/ca1_plane_pre_post_shear.py`)

A two-panel figure showing the first-canonical-angle plane between the
goal and no-goal residual subspaces of one of three configurations
(`--kind r | t | combined`).

The library underlying it -- `canonical_angles/ca1_plane.py` -- exposes:

* `CA1Decomposition` dataclass: the canonical vectors `a_1`, `b_1`, the
  bisectors `e_+`, `e_-` (eigenvectors of the orthogonalising shear),
  the shear eigenvalues `lambda_+ = sqrt(tan(theta_1/2))` and
  `lambda_- = sqrt(cot(theta_1/2))`, and the post-shear unit directions
  `a_1' = (e_+ + e_-) / sqrt(2)`, `b_1' = (e_+ - e_-) / sqrt(2)`.
* `compute_ca1_decomposition(A, B)`: the math, geometry-agnostic about
  what the subspaces are.
* `plot_ca1_plane_pre_post_shear(decomp, entities, ...)`: the figure.
  Pre-shear is a tall-narrow panel on the left (data hugs `e_+`); post-shear
  is a square panel on the right (data fills the plane after the shear
  redistribution).  All entities are projected into the plane,
  collision-aware-labelled by semantic priority + corpus membership, and
  the four corpus centroids (all roles, all traits, goal corpus, nogoal
  corpus) are overlaid as stars.  Returns a `matplotlib.figure.Figure`;
  the caller saves with appropriate metadata.

Example::

    # Combined pooled CA1 plane:
    uv run python -m results_analysis.canonical_angles.plots.ca1_plane_pre_post_shear \
        --kind combined --slot 3 --layer 25 \
        --output roger/ca1_plane_combined_s3_L25.png

    # r and t kinds (the 30-vs-30 single-grid versions):
    for kind in r t; do
        uv run python -m results_analysis.canonical_angles.plots.ca1_plane_pre_post_shear \
            --kind $kind --slot 3 --layer 25 \
            --output roger/ca1_plane_${kind}_s3_L25.png
    done

By default the plot uses theatricality-shifted residuals; pass
`--no-theat_shift` to disable.  The standalone-corpus colour-coding (red =
goal, blue = nogoal, grey = neither) reads `data/goal_roles_and_traits.json`
unless `--goal_list` overrides.

## Adding a new wrapper

A typical wrapper has the following structure:

```python
from results_analysis.canonical_angles import (
    CASpec, AggSpec, OriginSpec, WhiteningSpec,
    ComputeEnv, compute_ca_grid,
)
from results_analysis.canonical_angles.data import (
    build_subspace, build_whitening_pool, detect_n_slots,
)
from results_analysis.canonical_angles.plot_helpers import plot_per_slot_panels


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)

    # 1. Build the subspaces (default aggregation: combo_residual)
    sub_a = tuple(build_subspace(data_dir, args.kind, "goal", "combo_residual"))
    sub_b = tuple(build_subspace(data_dir, args.kind, "nogoal", "combo_residual"))

    # 2. Build whitening pool
    pool = tuple(build_whitening_pool(data_dir, scope="roles+traits"))

    # 3. Construct CASpecs over the variations
    specs = []
    for slot in args.slots:
        for w in args.whitening:
            specs.append(CASpec(
                subspace_a=sub_a, subspace_b=sub_b,
                slot_indices=(slot,), slot_mode="single",
                layer=args.layer,
                aggregation=AggSpec(mode="combo_residual"),
                origin=OriginSpec(kind="none", combos_kind=args.kind),
                whitening=WhiteningSpec(method=w[0], K=w[1], pool=pool),
            ))

    # 4. Compute (with caching)
    results = compute_ca_grid(specs, data_dir=data_dir)

    # 5. Build plot records and plot
    records = [...]
    fig = plot_per_slot_panels(records, panel_key="slot", group_key="whitening", ...)
    fig.savefig(args.output)
```

The compute layer caches loaded entity vectors (`(etype, name)`), fitted
whitening bases (`(method, K, pool, slot_label, layer)`), and computed
origins so a sweep over many specs with shared intermediates runs in
amortized constant time.

## Verifying outputs

To check that a new wrapper or change reproduces a historical plot, the
straightforward sanity check is:

1. Run the wrapper to produce a fresh PNG.
2. Visually compare to the historical PNG (open both side-by-side).
3. Compare the printed per-slot summary (min / median / max angle) to the
   historical values either by rerunning the historical script (if the code
   is available) or by inspecting the printed text in the plot file headers.

For the goal-vs-nogoal computation specifically, you can also load the
on-disk marginals and replicate the inner SVD step in a few lines for a
hard numerical comparison; see the verification done in the implementation
PR for an example.
