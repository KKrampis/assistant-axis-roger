#!/usr/bin/env python3
"""Find axes nearly orthogonal to the entire response-judged cohort.

For each axis in ``pair_list_clean.json`` that does NOT have response
judging (i.e. is not in ``pair_list_responses.json``), compute its
max ``|cos|`` against any axis that DOES have response judging.
Lists the orthogonal-to-response axes sorted ascending by that max,
groups them into singletons / pairs / trios / etc. based on mutual
``|cos| >= 0.50`` within the orthogonal set, and reports each
cluster's three closest response axes.

Reads the canonical datastore
``roger/axis_judge_experiments/axis_cosine_heatmap_clean_softshear3.json``
(L=3 soft-shear frame, slot 6, layer 25, ca_kind=combined).  Pure
JSON read + numpy lookup — sub-second runtime regardless of cohort
size.

Use case: identifying which non-response-judged axes would, if
response-judged, add the most NEW geometric coverage to the
response cohort.  An axis whose direction is already heavily covered
by the existing response axes is a low-marginal-yield candidate;
one whose direction is nearly orthogonal to the entire response
cohort is a high-marginal-yield candidate.

CLI::

    # Default: L=3 soft-shear frame, threshold 0.50
    uv run python results_analysis/orthogonal_to_response_set.py

    # Raw (L=0) frame for cross-frame comparison
    uv run python results_analysis/orthogonal_to_response_set.py --raw

    # Stricter cluster-membership threshold
    uv run python results_analysis/orthogonal_to_response_set.py --threshold 0.40
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

DEFAULT_EXPERIMENT_DIR = (
    Path(__file__).resolve().parent.parent / "roger" / "axis_judge_experiments"
)


def _union_find_components(
    edges: list[tuple[float, str, str]],
) -> dict[str, list[str]]:
    """Connected components from an undirected edge list.

    Returns a dict mapping representative -> list of nodes in that
    component.  Edge weights ignored (the edge set is what matters
    for connectivity).
    """
    members: set[str] = set()
    for _, a, b in edges:
        members.add(a)
        members.add(b)
    parent = {m: m for m in members}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for _, a, b in edges:
        union(a, b)

    groups: dict[str, list[str]] = defaultdict(list)
    for m in members:
        groups[find(m)].append(m)
    return groups


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--experiment_dir", type=Path, default=DEFAULT_EXPERIMENT_DIR,
        help=f"Directory containing the cosine datastore + pair lists "
             f"(default: {DEFAULT_EXPERIMENT_DIR}).",
    )
    p.add_argument(
        "--raw", action="store_true",
        help="Use the raw (L=0) cosine datastore instead of the "
             "canonical L=3 soft-shear datastore.",
    )
    p.add_argument(
        "--threshold", type=float, default=0.50,
        help="Max |cos| threshold for 'orthogonal to the response "
             "cohort' (default 0.50).  An axis qualifies as orthogonal "
             "iff its max |cos| against any response axis is below this.",
    )
    p.add_argument(
        "--cluster_threshold", type=float, default=0.50,
        help="Mutual |cos| threshold for grouping orthogonal axes "
             "into pairs/trios/etc. (default 0.50 to match the project "
             "convention used in axis_cosine_seriation.py's cluster "
             "bracket overlays).",
    )
    args = p.parse_args()

    fname = ("axis_cosine_heatmap_clean_raw.json" if args.raw
             else "axis_cosine_heatmap_clean_softshear3.json")
    print(f"Datastore: {fname}")
    d = json.load(open(args.experiment_dir / fname))
    r = d["result"]
    labels: list[str] = r["axis_labels"]
    M = np.array(r["abs_cos_matrix"])

    resp_pairs = json.load(open(args.experiment_dir / "pair_list_responses.json"))
    resp_axes = {f"{p['pos']}/{p['neg']}" for p in resp_pairs}
    resp_idx = {labels.index(a) for a in resp_axes if a in labels}
    non_resp_idx = sorted(set(range(len(labels))) - resp_idx)

    print(f"Cohort: {len(labels)} clean axes; "
          f"{len(resp_idx)} have response judging; "
          f"{len(non_resp_idx)} do not.\n")

    # For each non-response axis, compute max |cos| against any
    # response axis, plus the full ranked list.
    results = []
    for i in non_resp_idx:
        ranked = [(labels[j], float(M[i, j])) for j in resp_idx]
        ranked.sort(key=lambda kv: -kv[1])
        results.append((labels[i], i, ranked))
    results.sort(key=lambda t: t[2][0][1])  # ascending by max |cos|

    print(f"{'axis':45s}  {'max|cos|':>9s}  closest response axis")
    print("-" * 95)
    for axis, _, ranked in results:
        best_label, best = ranked[0]
        print(f"  {axis:43s}  {best:>9.3f}  {best_label}")

    # Filter to axes meeting the orthogonality threshold
    print(f"\n=== Tight clusters within axes with max|cos| < {args.threshold} ===")
    orth = [(a, i) for a, i, ranked in results
            if ranked[0][1] < args.threshold]
    print(f"{len(orth)} non-response axes meet the threshold.\n")

    # Build the mutual-|cos|>=cluster_threshold edge list within the
    # orthogonal subset.
    pair_links: list[tuple[float, str, str]] = []
    for a_idx, (a, i) in enumerate(orth):
        for b, j in orth[a_idx + 1:]:
            v = float(M[i, j])
            if v >= args.cluster_threshold:
                pair_links.append((v, a, b))
    pair_links.sort(reverse=True)

    print(f"Pairs with mutual |cos| >= {args.cluster_threshold}:")
    if pair_links:
        for v, a, b in pair_links:
            print(f"  |cos|={v:.3f}   {a}  <->  {b}")
    else:
        print("  (none)")

    # Connected components (clusters) in the edge list
    in_cluster: set[str] = set()
    if pair_links:
        groups = _union_find_components(pair_links)
        print("\nConnected components (tight clusters among orthogonal axes):")
        for members_list in sorted(groups.values(), key=lambda lst: -len(lst)):
            n_members = len(members_list)
            label_kind = ("Pair" if n_members == 2 else
                          "Trio" if n_members == 3 else
                          "Quartet" if n_members == 4 else
                          f"Group of {n_members}")
            print(f"\n  [{label_kind}] {sorted(members_list)}")
            in_cluster.update(members_list)
            # Also report what these cluster members are closest to
            # in the response cohort.
            for m in sorted(members_list):
                ranked = next(r[2] for r in results if r[0] == m)
                top = ranked[:3]
                top_s = "; ".join(f"{lbl} ({v:.3f})" for lbl, v in top)
                print(f"    {m}: nearest response axes -- {top_s}")

    # Singletons: orthogonal-to-response with no cluster-mutual-|cos|
    # buddy at the cluster threshold.
    singletons = [a for a, _ in orth if a not in in_cluster]
    print(f"\n=== Singletons (orthogonal to response, "
          f"no |cos| >= {args.cluster_threshold} buddy in the orth set) ===")
    for a in singletons:
        ranked = next(r[2] for r in results if r[0] == a)
        top = ranked[:3]
        top_s = "; ".join(f"{lbl} ({v:.3f})" for lbl, v in top)
        print(f"  {a}: max|cos|={ranked[0][1]:.3f}; nearest -- {top_s}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
