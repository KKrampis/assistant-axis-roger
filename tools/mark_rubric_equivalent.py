#!/usr/bin/env python3
"""Mark a rubric-version bump as prompt-preserving for a scope.

Sister CLI to ``tools/mark_script_equivalent.py``.  Where that one
declares "this producer-script edit was output-preserving" (so
downstream caches keep their fresh status across the edit), *this*
one declares "this rubric-version bump was prompt-preserving for
the following (axes, modes) cells" -- so the producer-side
:func:`_check_rubric_version_on_resume` will keep the cache
instead of rejudging.

Typical use after a rubric bump:

    # The v2 → v3 motivating example: response-mode prompts were
    # byte-identical for all but one axis (only systems_thinker has
    # a multi-word pole).  Declare the equivalence so we don't
    # rejudge the 66 response-mode v2 caches on the 11 single-pole
    # axes.
    uv run python tools/mark_rubric_equivalent.py \\
        --from v2 --to v3 \\
        --modes responses \\
        --except-axes systems_thinker_vs_analytical \\
        --reason "v3 differs from v2 only in display-form vs file-form name rendering; response-mode prompts are byte-identical for axes without multi-word pole names."

Symmetric declarations (``--from v2 --to v3`` and the reverse)
can be added in one call with ``--symmetric``; equivalence is
mathematically symmetric but the registry stores directional edges
because some callers ask "is the cached rubric ≤ current?".

To inspect existing edges:

    uv run python tools/mark_rubric_equivalent.py --list
    uv run python tools/mark_rubric_equivalent.py --list \\
        --from v2 --to v3

To check whether a specific edge is recognised for a (axis, mode)
cell (without mutating the registry):

    uv run python tools/mark_rubric_equivalent.py --check \\
        --from v2 --to v3 \\
        --axis concise_vs_verbose --mode responses
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Sequence

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from assistant_axis import rubric_equivalence as eq  # noqa: E402


def _print_list(
    registry: Sequence[eq.RubricEquivalenceEdge],
    *,
    from_rubric: Optional[str],
    to_rubric: Optional[str],
) -> None:
    rows = [
        e for e in registry
        if (from_rubric is None or e.from_rubric == from_rubric)
        and (to_rubric is None or e.to_rubric == to_rubric)
    ]
    if not rows:
        print("(no matching equivalence edges)")
        return
    for e in rows:
        scope_bits: List[str] = []
        if e.modes is not None:
            scope_bits.append(f"modes={list(e.modes)}")
        if e.axes is not None:
            scope_bits.append(f"axes={list(e.axes)}")
        if e.except_axes is not None:
            scope_bits.append(f"except_axes={list(e.except_axes)}")
        if e.entity_ids is not None:
            scope_bits.append(f"entity_ids={list(e.entity_ids)}")
        if e.except_entity_ids is not None:
            scope_bits.append(
                f"except_entity_ids={list(e.except_entity_ids)}"
            )
        scope = (", ".join(scope_bits) if scope_bits
                 else "(all axes, all modes, all entities)")
        print(f"{e.from_rubric} -> {e.to_rubric}  [{scope}]")
        if e.reason:
            print(f"    reason:    {e.reason}")
        if e.marked_at:
            print(f"    marked_at: {e.marked_at}")
        print()


def _parse_scope_lists(
    args: argparse.Namespace,
) -> tuple[
    Optional[List[str]], Optional[List[str]],
    Optional[List[str]], Optional[List[str]],
]:
    """Return (axes, except_axes, entity_ids, except_entity_ids) from the
    parser, enforcing mutual exclusion within each dimension."""
    axes = args.axes if args.axes else None
    except_axes = args.except_axes if args.except_axes else None
    entity_ids = args.entity_ids if args.entity_ids else None
    except_entity_ids = (args.except_entity_ids
                         if args.except_entity_ids else None)
    if axes and except_axes:
        raise SystemExit(
            "error: pass at most one of --axes (allow-list) or "
            "--except-axes (deny-list)."
        )
    if entity_ids and except_entity_ids:
        raise SystemExit(
            "error: pass at most one of --entity-ids (allow-list) or "
            "--except-entity-ids (deny-list)."
        )
    return axes, except_axes, entity_ids, except_entity_ids


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--from", dest="from_rubric", default=None,
                   help="Source rubric-version label (e.g. v2).")
    p.add_argument("--to", dest="to_rubric", default=None,
                   help="Target rubric-version label (e.g. v3).")
    p.add_argument("--modes", nargs="*", default=None,
                   help="Mode scope: subset of {descriptions, instructions, "
                        "responses}.  Omit to cover all modes.")
    p.add_argument("--axes", nargs="*", default=None,
                   help="Axis allow-list (mutually exclusive with "
                        "--except-axes).")
    p.add_argument("--except-axes", dest="except_axes", nargs="*",
                   default=None,
                   help="Axis deny-list (mutually exclusive with --axes).")
    p.add_argument("--entity-ids", dest="entity_ids", nargs="*",
                   default=None,
                   help="Entity-id allow-list (e.g. 'patient|R'); mutually "
                        "exclusive with --except-entity-ids.  Use disambiguated "
                        "ids ('name|R' / 'name|T'), not bare names: the 9 "
                        "trait/role collisions can't be expressed unambiguously "
                        "otherwise.  When a rubric bump affects only specific "
                        "entities (e.g. v2\u2192v3 differs only for entity names "
                        "containing underscores), enumerate them here so the "
                        "producer's per-entity rubric-drift check can filter "
                        "accurately rather than dropping the entire cohort.")
    p.add_argument("--except-entity-ids", dest="except_entity_ids", nargs="*",
                   default=None,
                   help="Entity-id deny-list; mutually exclusive with "
                        "--entity-ids.  Use the same 'name|R'/'name|T' format.")
    p.add_argument("--reason", default="",
                   help="Free-text justification recorded with the edge.")
    p.add_argument("--symmetric", action="store_true",
                   help="Also declare the reverse edge (to -> from) in "
                        "one shot.  Use when the rubric bump is genuinely "
                        "round-trippable (the common case).")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--list", dest="do_list", action="store_true",
                      help="List existing edges (optionally filtered by "
                           "--from / --to).")
    mode.add_argument("--check", dest="do_check", action="store_true",
                      help="Check whether (from, to) is equivalent for the "
                           "given --axis / --mode cell; exit 0 if yes, 1 "
                           "if no.  Does not mutate the registry.")
    p.add_argument("--axis", default=None,
                   help="Axis to use for --check (omit to test the axis "
                        "wildcard).")
    p.add_argument("--mode", default=None,
                   help="Mode to use for --check (omit to test the mode "
                        "wildcard).")
    p.add_argument("--entity-id", dest="entity_id", default=None,
                   help="Entity-id ('name|R' / 'name|T') to use for --check "
                        "(omit to test the entity wildcard).")
    args = p.parse_args(argv)

    registry = eq.load_registry()

    if args.do_list:
        _print_list(registry,
                    from_rubric=args.from_rubric,
                    to_rubric=args.to_rubric)
        return 0

    if args.do_check:
        if not (args.from_rubric and args.to_rubric):
            raise SystemExit("--check requires both --from and --to.")
        found, path = eq.is_equivalent(
            args.from_rubric, args.to_rubric,
            axis=args.axis, mode=args.mode, entity_id=args.entity_id,
            registry=registry, return_path=True,
        )
        scope = (f"axis={args.axis or '*'}, mode={args.mode or '*'}, "
                 f"entity_id={args.entity_id or '*'}")
        if found:
            if not path:
                print(f"YES (reflexive, {scope}): "
                      f"{args.from_rubric} == {args.to_rubric}")
            else:
                hops = " -> ".join(
                    [args.from_rubric] + [e.to_rubric for e in path]
                )
                print(f"YES ({scope}): {hops}")
            return 0
        print(f"NO ({scope}): {args.from_rubric} -> {args.to_rubric} "
              f"not declared.")
        return 1

    # Default action: append (one or two edges).
    if not (args.from_rubric and args.to_rubric):
        raise SystemExit(
            "error: --from and --to are required when appending an edge "
            "(or use --list / --check to query)."
        )
    if not args.reason:
        raise SystemExit(
            "error: --reason is required when appending an edge "
            "(human-readable justification for the audit trail)."
        )
    axes, except_axes, entity_ids, except_entity_ids = _parse_scope_lists(args)

    try:
        edge = eq.append_equivalence(
            from_rubric=args.from_rubric,
            to_rubric=args.to_rubric,
            modes=args.modes,
            axes=axes,
            except_axes=except_axes,
            entity_ids=entity_ids,
            except_entity_ids=except_entity_ids,
            reason=args.reason,
        )
    except ValueError as exc:
        raise SystemExit(f"error: {exc}") from exc
    print(f"declared: {edge.from_rubric} -> {edge.to_rubric}")

    if args.symmetric and edge.from_rubric != edge.to_rubric:
        try:
            rev = eq.append_equivalence(
                from_rubric=args.to_rubric,
                to_rubric=args.from_rubric,
                modes=args.modes,
                axes=axes,
                except_axes=except_axes,
                entity_ids=entity_ids,
                except_entity_ids=except_entity_ids,
                reason=args.reason + "  (reverse direction; --symmetric).",
            )
        except ValueError as exc:
            print(f"warning: reverse edge not added ({exc})")
        else:
            print(f"declared: {rev.from_rubric} -> {rev.to_rubric}  (reverse)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
