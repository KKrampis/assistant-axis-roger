#!/usr/bin/env python3
"""Pick base-persona role candidates for a full per-axis steering sweep.

For each axis in ``pair_list_clean.json`` (or another pair list),
produce 3-5 candidate roles to use as the BASE PERSONA when steering
along that axis.  The selection satisfies, in order of priority:

1. **Goal-pairing**: if the axis is *goal* → base role is *non-goal*;
   if the axis is *non_goal* or *None/mixed* → base role is *goal*.
   (Per Roger's rule, May 2026.)
2. **Real-roles preferred**: roles that are "a trait pretending to be a
   role" (``altruist``, ``hedonist``, ``narcissist`` etc.) are excluded;
   real social/professional/mythic roles are kept.
3. **Semi-human + verbal**: roles like ``virus``, ``predator``, ``void``,
   ``swarm``, ``toddler``, ``infant`` are excluded.
4. **Not a pole of the axis**: a role can't be its own steering base.
5. **Neutral on this axis**: rank by ``|mean(score across desc+inst,
   gpt+sonnet)|`` and prefer roles near 0 (avoid the "callous counsellor"
   problem where the base role is already strongly leaning in the
   steering direction).
6. **Variety**: penalise roles that have already been picked for many
   other axes so the same role isn't suggested 30 times.

Output: a plain-text file at ``--out`` (default
``roger/axis_judge_experiments/base_persona_candidates.txt``) with one
line per axis::

    <pos>/<neg>  [<goal_type>]  →  role1, role2, role3, role4, role5

Roger edits this file down to the final 2 roles per axis before the
sweep runs.

The script reads existing axis-judge correlation outputs (``scores_*.json``
under each axis directory) for the neutrality ranking; it does NOT call
any judge.  Runtime is a few seconds.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

DEFAULT_EXPERIMENT_DIR = REPO / "roger" / "axis_judge_experiments"
DEFAULT_OUT = DEFAULT_EXPERIMENT_DIR / "base_persona_candidates.txt"

# Hard exclude lists - hand-curated May 2026 per Roger's selection rules.
# These are EXCLUDED from the candidate pool regardless of goal-type.

# 1. Trait-as-role aliases: roles whose name is essentially a personality
# trait dressed up as a role identity, with no real-world social context.
TRAIT_AS_ROLE_EXCLUDE = {
    "altruist", "ascetic", "hedonist", "idealist", "narcissist",
    "pacifist", "optimist", "perfectionist", "workaholic", "daredevil",
    "contrarian", "romantic", "skeptic", "stoic", "realist",
    "pragmatist", "minimalist", "purist", "cynic", "dreamer",
    "loner", "nomad",
    # These are mostly clean professions / social roles BUT they
    # function as trait-as-role here because they're defined by a
    # single dispositional axis -- Roger wants real social/professional
    # roles whose identity isn't a trait label.
}

# 2. Non-verbal / non-human / collective / abstract roles.  These either
# can't speak in any natural sense or aren't an individual perspective.
NON_HUMAN_OR_NONVERBAL_EXCLUDE = {
    # animals / non-verbal lifeforms
    "predator", "prey", "parasite", "symbiont", "virus", "whale",
    "tree", "coral_reef", "mycorrhizal", "swarm", "hive", "ecosystem",
    "leviathan", "familiar",
    # abstract concepts / forces / collectives
    "void", "wind", "zeitgeist", "egregore", "echo", "aberration",
    "chimera", "homunculus", "crystalline", "hybrid",
    # eldritch / non-comprehensible
    "eldritch",
    # pre-verbal humans
    "toddler", "infant",
    # other low-utility
    "amnesiac",  # explicitly no memory -> hard to roleplay coherently
}

# 3. Roles whose identity is too vague/abstract for a meaningful base.
VAGUE_OR_ABSTRACT_EXCLUDE = {
    "simulacrum", "shapeshifter", "tulpa", "avatar", "default",
    "ancient",
}

# 4. Single-axis-extreme roles -- these are real archetypes but
# strongly pre-loaded on a major axis (alignment, etc).  Excluded
# because they'd give "callous counsellor"-style steering problems on
# any axis adjacent to that one.
SINGLE_AXIS_EXTREME_EXCLUDE = {
    "saint",  # ultra-positive on alignment axes
    "virtue_aligned_ai",  # ditto (added 2026-09-28 beside the instrumental conception)
    "criminal",  # ultra-negative on alignment
    "smuggler", "pirate", "rogue",  # ditto, milder
    "vegan",  # strong on ecocentric/spiritual-ish
    "luddite",  # strong on innovative/traditional
    "evangelist",  # strong on religious/secular and confident/uncertain
    "zealot", "fundamentalist",  # strong on absolutist axes
}

# 5. Roles that are POLES of axes in pair_list_clean.  By construction
# these are extreme on one or more steering axes (we picked them
# *because* they're far apart on a meaningful direction), so using
# them as a NEUTRAL base persona is contradictory.  These are
# excluded from EVERY axis's candidate pool, not just their own.
ROLE_POLE_EXCLUDE = {
    "aligned_artificial_intelligence", "paperclip_maximizer",
    # renamed 2026-09-28; the old stem stays for data extracted before then
    "instrumentally_aligned_ai",
    "angel", "demon",
    "guardian", "destroyer",
    "symbiont", "parasite",
    "predator", "prey",
    # Some of these (predator/prey/parasite/symbiont) are already
    # excluded under NON_HUMAN_OR_NONVERBAL_EXCLUDE; listed redundantly
    # for documentation completeness.
}

# Union of all exclusions.
HARD_EXCLUDE = (
    TRAIT_AS_ROLE_EXCLUDE
    | NON_HUMAN_OR_NONVERBAL_EXCLUDE
    | VAGUE_OR_ABSTRACT_EXCLUDE
    | SINGLE_AXIS_EXTREME_EXCLUDE
    | ROLE_POLE_EXCLUDE
)


def _safe_load_json(p: Path) -> dict | None:
    if not p.exists():
        return None
    try:
        d = json.loads(p.read_text())
        return d.get("result", d)  # provenance envelope or flat
    except Exception:
        return None


def load_axis_role_scores(
    axis_dir: Path,
    known_roles: set[str],
    known_traits: set[str],
) -> dict[str, list[int]]:
    """Return ``{role_name: [score, score, ...]}`` for all roles found
    in this axis's score files (gpt + sonnet, desc + inst).

    Handles both schemas:

    * v2 (post-May 2026 disambiguator): keys are ``"name|R"`` /
      ``"name|T"`` — filter on ``|R`` suffix.
    * v1 (pre-disambiguator): keys are bare names.  Cross-reference
      against the operative-corpus role/trait sets to figure out which
      kind each bare key represents; drop bare names that are
      ambiguous (in both kinds) since v1 silently overwrote one side.
      Mirrors ``migrate_v1_static_scores`` in
      ``assistant_axis.judge_loaders``.
    """
    out: dict[str, list[int]] = defaultdict(list)
    for judge in ("gpt", "sonnet"):
        for mode in ("descriptions", "instructions"):
            d = _safe_load_json(axis_dir / judge / f"scores_{mode}.json")
            if not d:
                continue
            for eid, score in d.items():
                if not isinstance(score, int):
                    continue
                if "|" in eid:
                    # v2 form: filter to roles only.
                    if not eid.endswith("|R"):
                        continue
                    role = eid[:-2]
                else:
                    # v1 bare-name: classify by corpus membership.
                    if eid in known_roles and eid in known_traits:
                        # Collision: v1 silently overwrote, can't trust.
                        continue
                    if eid not in known_roles:
                        continue
                    role = eid
                out[role].append(score)
    return out


def axis_pos_neg_to_dirname(pos: str, neg: str) -> str:
    return f"{pos}_vs_{neg}"


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--pairs", default="pair_list_clean.json",
                   help="Pair list to process (default: pair_list_clean.json).")
    p.add_argument("--experiment_dir", default=str(DEFAULT_EXPERIMENT_DIR))
    p.add_argument("--out", default=str(DEFAULT_OUT))
    p.add_argument("--n_candidates", type=int, default=5,
                   help="Max candidates per axis (default: 5).")
    p.add_argument("--min_judges", type=int, default=2,
                   help="Min number of judge scores per (role,axis) to "
                        "consider the role's neutrality estimate reliable "
                        "(default: 2; out of max 4 = 2 modes × 2 judges).")
    p.add_argument("--variety_penalty", type=float, default=0.35,
                   help="Per-prior-pick penalty added to |mean_score| when "
                        "ranking candidates (default: 0.35; 0 disables "
                        "the variety-spreading mechanic).  At 0.35 a role "
                        "that's already been picked once is treated as if "
                        "its |mean_score| had increased by 0.35 -- enough "
                        "to lose to a fresh role at |μ|=0 but to win over "
                        "a role at |μ|>=0.5.  The goal pool is fairly "
                        "thin after exclusions (~22 roles) and is drawn "
                        "from by ~38 axes, so a strong variety push is "
                        "needed to spread role usage.")
    p.add_argument("--goal_roles_traits",
                   default="data/goal_roles_and_traits.json",
                   help="Goal/non-goal classification file "
                        "(default: data/goal_roles_and_traits.json).")
    args = p.parse_args()

    experiment_dir = Path(args.experiment_dir).resolve()
    pairs_path = experiment_dir / args.pairs
    pairs = json.loads(pairs_path.read_text())
    print(f"Loaded {len(pairs)} axes from {pairs_path}")

    # ----- Build role pool, classified by goal_type -----
    g = json.loads((REPO / args.goal_roles_traits).read_text())
    goal_roles = set(g["roles"]["goal"])
    non_goal_roles = set(g["roles"]["non_goal"])
    role_goal_type: dict[str, str] = {}
    for r in goal_roles:
        role_goal_type[r] = "goal"
    for r in non_goal_roles:
        role_goal_type[r] = "non_goal"

    # Also collect roles that are POLES of axes in the current pair list;
    # those should never be used as a base for THEIR OWN axis (handled
    # per-axis below) -- but they're fine for OTHER axes.
    pole_roles_per_axis: dict[tuple[str, str], set[str]] = {}
    for it in pairs:
        if it.get("pair_type") == "roles":
            pole_roles_per_axis[(it["pos"], it["neg"])] = {it["pos"], it["neg"]}
        else:
            pole_roles_per_axis[(it["pos"], it["neg"])] = set()

    print(f"Goal roles: {len(goal_roles)}, non-goal: {len(non_goal_roles)}")
    print(f"Hard-excluded (trait-as-role / non-human / abstract / single-axis-"
          f"extreme): {len(HARD_EXCLUDE)}")
    eligible_goal = sorted(goal_roles - HARD_EXCLUDE)
    eligible_non_goal = sorted(non_goal_roles - HARD_EXCLUDE)
    print(f"  → eligible goal roles: {len(eligible_goal)}")
    print(f"  → eligible non-goal roles: {len(eligible_non_goal)}")

    # Build name → kind disambiguator from the corpus layout.  v1 score
    # caches use bare names; we need to know which kind each name maps to.
    all_roles_in_corpus = set(goal_roles) | set(non_goal_roles)
    all_traits_in_corpus = set(g["traits"]["goal"]) | set(g["traits"]["non_goal"])

    # ----- Build the per-axis neutrality matrix -----
    axis_role_score: dict[tuple[str, str], dict[str, float]] = {}
    axis_role_count: dict[tuple[str, str], dict[str, int]] = {}
    missing_dirs = 0
    for it in pairs:
        axis_key = (it["pos"], it["neg"])
        axis_dir = experiment_dir / axis_pos_neg_to_dirname(it["pos"], it["neg"])
        if not axis_dir.exists():
            missing_dirs += 1
            axis_role_score[axis_key] = {}
            axis_role_count[axis_key] = {}
            continue
        role_scores = load_axis_role_scores(
            axis_dir, all_roles_in_corpus, all_traits_in_corpus,
        )
        mean_score = {r: sum(s) / len(s) for r, s in role_scores.items() if s}
        score_count = {r: len(s) for r, s in role_scores.items() if s}
        axis_role_score[axis_key] = mean_score
        axis_role_count[axis_key] = score_count
    print(f"Loaded score data for {len(pairs) - missing_dirs}/"
          f"{len(pairs)} axes ({missing_dirs} dirs missing or empty)")

    # ----- Greedy candidate selection with variety penalty -----
    pick_counts: Counter = Counter()
    lines: list[str] = []
    for it in pairs:
        axis_key = (it["pos"], it["neg"])
        goal_type = it.get("goal_type")
        # Roger's rule: goal axis → non-goal role; else → goal role.
        target_pool_name = "non_goal" if goal_type == "goal" else "goal"
        target_pool = (
            eligible_non_goal if target_pool_name == "non_goal"
            else eligible_goal
        )
        # Exclude this axis's own poles (if it's a role-pair).
        own_poles = pole_roles_per_axis[axis_key]

        scores = axis_role_score[axis_key]
        counts = axis_role_count[axis_key]

        # Score each eligible role: |mean_score| + variety_penalty * times_picked.
        candidates: list[tuple[float, float, int, str]] = []
        for role in target_pool:
            if role in own_poles:
                continue
            if role not in scores:
                # No score data on this axis for this role -- skip rather
                # than guess.  In practice this only happens for roles
                # that weren't in the corpus when this axis was judged
                # (rare; the corpus is pretty stable).
                continue
            if counts.get(role, 0) < args.min_judges:
                continue
            mean = scores[role]
            picked = pick_counts[role]
            rank_score = abs(mean) + args.variety_penalty * picked
            candidates.append((rank_score, abs(mean), picked, role))

        no_scores_fallback = False
        if not candidates and not scores:
            # No judge data for this axis at all (likely still pending in
            # the running di-extend judging batch).  Fall back to picking
            # the least-used roles from the pool so Roger has SOMETHING to
            # consider; flag it loudly so he treats those as unverified.
            no_scores_fallback = True
            fallback = [
                (pick_counts[r], r) for r in target_pool
                if r not in own_poles
            ]
            fallback.sort()
            chosen = [r for _, r in fallback[: args.n_candidates]]
        else:
            candidates.sort()
            chosen = [r for _, _, _, r in candidates[: args.n_candidates]]
        for r in chosen:
            pick_counts[r] += 1

        # One section per axis (markdown-ish ## header + indented
        # bullets, one candidate per line) -- format chosen for
        # easy editing: delete the lines you don't want, the
        # remaining bullets are your final picks.  See the file
        # header for the full instructions.
        gt_tag = goal_type if goal_type else "mixed"
        flag = "  [NO SCORES -- judging pending]" if no_scores_fallback else ""
        lines.append(
            f"## {it['pos']}/{it['neg']}  "
            f"[axis={gt_tag}, base from {target_pool_name} pool]{flag}"
        )
        if not chosen:
            lines.append("  (no candidates)")
        else:
            # Pad role names to align the (|μ|, n) annotations.
            max_role_len = max(len(r) for r in chosen)
            for r in chosen:
                if no_scores_fallback:
                    suffix = "(no score data)"
                else:
                    suffix = f"(|μ|={abs(scores[r]):.2f}, n={counts[r]})"
                lines.append(f"  {r:<{max_role_len}}  {suffix}")
        lines.append("")  # blank line separator

    # ----- Write output -----
    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "# Base-persona candidates for full per-axis steering sweep",
        f"# Generated: {Path(__file__).name} from {pairs_path.name}",
        "#",
        "# HOW TO USE:",
        "#   1. Scan each ## axis section below.",
        "#   2. Decide which 2 (or however many) candidates you want as",
        "#      base personas for that axis's steering sweep.",
        "#   3. DELETE the lines for the candidates you don't want.",
        "#   4. The remaining bullets under each axis are your final",
        "#      picks; downstream tooling can grep for indented role",
        "#      names per ## section.",
        "#",
        "# Rules applied during generation:",
        "#  - axis goal_type=goal      → base role from NON-GOAL pool",
        "#  - axis goal_type=non_goal  → base role from GOAL pool",
        "#  - axis goal_type=None/mixed → base role from GOAL pool",
        "#  - exclude trait-as-role aliases (altruist, narcissist, ...)",
        "#  - exclude non-verbal / non-human / abstract (virus, void, ...)",
        "#  - exclude single-axis-extremes (saint, criminal, evangelist, ...)",
        "#  - exclude role's own poles when the axis is a role-pair",
        "#  - rank by |mean judge score| across desc+inst, gpt+sonnet;",
        "#    lower = more neutral on this axis = better base",
        "#  - variety penalty (default 0.35) added per prior pick to",
        "#    spread roles across axes",
        "#",
        "# Annotations on each bullet:",
        "#   |μ| = absolute mean judge score on this axis",
        "#         (0 = perfectly neutral, 3 = extreme)",
        "#   n   = how many of the 4 max judge scores (2 judges × 2",
        "#         modes) were available for this (axis, role) pair",
        "",
    ]
    out_path.write_text("\n".join(header + lines) + "\n")
    print(f"Wrote {out_path}")

    # Print top-10 most-picked roles for sanity (variety check)
    top = pick_counts.most_common(10)
    print(f"\nTop 10 most-picked roles "
          f"(variety-penalty={args.variety_penalty}):")
    for r, c in top:
        print(f"  {r:<20} picked {c}× across {len(pairs)} axes")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
