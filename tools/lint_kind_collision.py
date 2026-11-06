#!/usr/bin/env python3
"""Lint helper for the trait/role name-collision footgun ("Bug A").

Walks one or more Python files (or directories) looking for the
canonical dual-iteration pattern that triggered Bug A in 14
consumer files:

    for et in ("traits", "roles"):
        ...
        somedict[name] = ...      # ← bare-name key: collision risk!

Reports each occurrence as a one-line ``<path>:<line>: <pattern>``
diagnostic with a suggestion to switch to ``entity_id(name, et)``
(see :mod:`assistant_axis.entity_id`).

This is a heuristic AST-based linter — it can't reason about whether
a particular dict assignment is *actually* mixed-kind, so it errs
toward false positives.  Use ``# noqa: kind-collision`` on a line to
silence a false alarm (or, better, refactor to use ``entity_id``
explicitly so the collision-safety is local + obvious).

Exit code:
* 0 = no findings (or only suppressed ones)
* 1 = at least one un-suppressed finding (suitable for CI gate)

Usage::

    uv run python tools/lint_kind_collision.py results_analysis/
    uv run python tools/lint_kind_collision.py path/to/file.py
    uv run python tools/lint_kind_collision.py            # walks repo

The ``--strict`` flag treats heuristic matches inside test files
as failures too (default: tests are exempt — they often deliberately
demonstrate the buggy pattern as part of regression coverage).
"""
from __future__ import annotations

import argparse
import ast
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional


_NOQA_TAG = "noqa: kind-collision"

# Directories to skip when walking the repo by default.  Tests are
# scanned but their findings are demoted to warnings unless --strict
# is passed (test code is allowed to demonstrate the bug pattern in
# its regression-test reproducer fixtures).
_SKIP_DIRS = {
    ".git", ".venv", ".pytest_cache", "__pycache__",
    "node_modules", ".mypy_cache",
    "archive",            # historical snapshots — frozen
    "runpod_workspace",   # large dataset directory
    "assistant_axis_jl",  # Julia experiments — N/A
    "data",               # not Python
    "roger",              # cache fan-out tree, not source
}

# Filenames inside `tests/` directories where the bug-pattern is
# expected (regression reproducers).  Findings here become warnings
# instead of errors unless --strict is set.
_TEST_PATH_FRAGMENTS = ("/tests/", "/test_")


@dataclass(frozen=True)
class Finding:
    path: Path
    lineno: int
    pattern: str   # short label (e.g. "dual-iter-bare-key")
    message: str   # one-line human description with suggestion
    is_test: bool


def _is_kind_tuple_literal(node: ast.AST) -> bool:
    """``("traits", "roles")`` or ``("roles", "traits")`` — the
    canonical Bug-A iterator.  Order doesn't matter."""
    if not isinstance(node, ast.Tuple):
        return False
    if len(node.elts) != 2:
        return False
    elts = node.elts
    string_values = []
    for e in elts:
        if isinstance(e, ast.Constant) and isinstance(e.value, str):
            string_values.append(e.value)
        else:
            return False
    return set(string_values) == {"traits", "roles"}


def _name_used_as_key(target: ast.AST, var_names: set[str]) -> bool:
    """Returns True iff ``target`` looks like ``somedict[name]``
    where ``name`` is one of the loop's name-binding variables and
    NOT obviously already an entity_id (e.g. ``entity_id(name, et)``
    or a slice).
    """
    if not isinstance(target, ast.Subscript):
        return False
    sl = target.slice
    # Python <3.9 wraps slices in ast.Index; Python 3.9+ uses the
    # expression directly.
    expr = sl.value if isinstance(sl, ast.Index) else sl  # type: ignore[attr-defined]
    if isinstance(expr, ast.Name) and expr.id in var_names:
        return True
    return False


def _file_contains_noqa_on_line(source_lines: list[str], lineno: int) -> bool:
    if 1 <= lineno <= len(source_lines):
        return _NOQA_TAG in source_lines[lineno - 1]
    return False


class _Visitor(ast.NodeVisitor):
    """AST visitor that flags the dual-iteration + bare-name-key
    footgun.

    Heuristic: when we enter a ``for et in ("traits", "roles"):``
    loop, track common short var names that often hold bare entity
    names inside the loop (``name``, ``n``, ``stem``, ``etname``,
    plus any variable assigned from ``fp.stem`` or
    ``Path(...).stem``).  Any subscript assignment ``D[<one-of-those>]
    = ...`` is reported.
    """

    # Variable names that, inside a kind-iteration loop, almost
    # always hold a bare entity name.  Heuristic; widen if FN-rate
    # turns out high.
    _BARE_NAME_VARS_BASE = frozenset({
        "name", "n", "stem", "etname", "ent_name", "entity_name",
    })

    def __init__(self, path: Path, source_lines: list[str]) -> None:
        self.path = path
        self.source_lines = source_lines
        self.findings: List[Finding] = []
        self.is_test = any(frag in str(path) for frag in _TEST_PATH_FRAGMENTS)
        self._loop_var_stack: list[set[str]] = []  # active kind-loop vars

    def _push_kind_loop(self, et_var: Optional[str]) -> None:
        names = set(self._BARE_NAME_VARS_BASE)
        # The kind-iter variable itself isn't a bare-name key target,
        # but we'll skip flagging assignments to D[et] (which would
        # be {et: ...} and is fine).
        self._loop_var_stack.append(names)

    def _pop_kind_loop(self) -> None:
        if self._loop_var_stack:
            self._loop_var_stack.pop()

    def visit_For(self, node: ast.For) -> None:
        is_kind_loop = _is_kind_tuple_literal(node.iter)
        if is_kind_loop:
            et_var = node.target.id if isinstance(node.target, ast.Name) else None
            self._push_kind_loop(et_var)
        try:
            self.generic_visit(node)
        finally:
            if is_kind_loop:
                self._pop_kind_loop()

    def visit_Assign(self, node: ast.Assign) -> None:
        if self._loop_var_stack:
            active_vars = self._loop_var_stack[-1]
            for target in node.targets:
                if _name_used_as_key(target, active_vars):
                    if not _file_contains_noqa_on_line(
                        self.source_lines, node.lineno
                    ):
                        self.findings.append(Finding(
                            path=self.path,
                            lineno=node.lineno,
                            pattern="dual-iter-bare-key",
                            message=(
                                "dual ('traits', 'roles') iteration with "
                                "bare-name dict key: collision-prone (Bug A). "
                                "Use entity_id(name, et) instead."
                            ),
                            is_test=self.is_test,
                        ))
        self.generic_visit(node)


def lint_file(path: Path) -> List[Finding]:
    """Lint a single Python file.  Syntax errors are silently
    skipped (the lint isn't trying to be a parser)."""
    try:
        source = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return []
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError:
        return []
    visitor = _Visitor(path, source.splitlines())
    visitor.visit(tree)
    return visitor.findings


def _iter_python_files(roots: Iterable[Path]) -> Iterable[Path]:
    for root in roots:
        root = Path(root)
        if root.is_file() and root.suffix == ".py":
            yield root
            continue
        if not root.is_dir():
            continue
        for p in root.rglob("*.py"):
            if any(part in _SKIP_DIRS for part in p.parts):
                continue
            yield p


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.strip().splitlines()[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "paths", nargs="*",
        help="Files or directories to lint (default: repo root).",
    )
    parser.add_argument(
        "--strict", action="store_true",
        help="Treat findings inside test files as errors (default: warnings).",
    )
    args = parser.parse_args(argv)

    if args.paths:
        roots = [Path(p) for p in args.paths]
    else:
        roots = [Path(__file__).resolve().parent.parent]

    findings: list[Finding] = []
    for f in _iter_python_files(roots):
        findings.extend(lint_file(f))

    n_errors = 0
    n_warnings = 0
    for f in findings:
        is_warning = f.is_test and not args.strict
        kind = "warning" if is_warning else "error"
        print(f"{f.path}:{f.lineno}: {kind}: [{f.pattern}] {f.message}")
        if is_warning:
            n_warnings += 1
        else:
            n_errors += 1
    n = len(findings)
    print(
        f"\n[lint_kind_collision] {n} finding(s): "
        f"{n_errors} error, {n_warnings} warning "
        f"(silence with '# {_NOQA_TAG}')",
        file=sys.stderr,
    )
    return 1 if n_errors > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
