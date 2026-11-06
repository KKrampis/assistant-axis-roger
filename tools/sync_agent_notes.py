#!/usr/bin/env python3
"""Generate the Claude Code instruction files from ``AGENT_NOTES.md``.

``AGENT_NOTES.md`` is the canonical, hand-maintained collaboration guide
for every AI agent working on this repo (Cursor reads it directly).
Claude Code, however, only auto-loads ``CLAUDE.md`` plus the rule files
under ``.claude/rules/`` (path-scoped: a rule loads when Claude reads or
edits a file matching its ``paths`` globs, and is restored after context
compaction) and the skills under ``.claude/skills/``.  Hand-maintained
copies would drift, so this script derives all of them from invisible
HTML-comment markers inside ``AGENT_NOTES.md``:

* One **targets block** near the top defines the rule and skill names::

      <!-- claude-sync
      rules:
        plotting:
          when: generating or verifying any plot
          paths: ["results_analysis/**/*.py"]
        working-style:
          when: every session        # no paths => always loaded
      skills:
        steering-to-gsheet:
          description: Export a steering experiment to Google Sheets
      -->

* One **section marker** on the line after a heading routes that section
  and, by inheritance, every sub-heading that has no marker of its own::

      ## Some heading
      <!-- claude: always -->            -> CLAUDE.md
      <!-- claude: rule=plotting -->     -> .claude/rules/plotting.md
      <!-- claude: skill=NAME -->        -> .claude/skills/NAME/SKILL.md
      <!-- claude: archive -->           -> stays only in AGENT_NOTES.md

  Unmarked top-level (``##``) sections default to ``archive`` and are
  reported, so new material can neither silently bloat context nor
  silently vanish from Claude Code.  ``CLAUDE.md`` additionally gets a
  generated rule map ("when doing X, read rule Y"), the skill list, and
  an index of the archive-only sections.

Generated files carry a banner line; the script refuses to overwrite a
file that lacks it (use ``--force``) and removes banner-carrying files
that no longer correspond to a target.

Exit code:
* 0 = generated files are up to date (or were just regenerated)
* 1 = ``--check`` found stale, missing, or orphaned generated files
* 2 = marker / targets-block error (unknown rule name, stray marker,
  hand-written file in the way, ...)

Usage::

    uv run python tools/sync_agent_notes.py            # regenerate
    uv run python tools/sync_agent_notes.py --check    # CI / pre-commit gate
    uv run python tools/sync_agent_notes.py --quiet    # session-start hook:
                                                       # prints only when
                                                       # something changed
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import yaml

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from assistant_axis.atomic_io import atomic_write_text  # noqa: E402

SOURCE_NAME = "AGENT_NOTES.md"
CLAUDE_MD = "CLAUDE.md"
RULES_DIR = ".claude/rules"
SKILLS_DIR = ".claude/skills"
SYNC_CMD = "uv run python tools/sync_agent_notes.py"
# NB: no "--" inside the comment body; HTML comments must not contain it.
BANNER = (
    f"<!-- GENERATED FILE: do not edit.  Source: {SOURCE_NAME} "
    f"(section markers).  Regenerate with: {SYNC_CMD} -->"
)

_TARGETS_OPEN_RE = re.compile(r"^<!--\s*claude-sync\s*$")
_COMMENT_CLOSE_RE = re.compile(r"^\s*-->\s*$")
_MARKER_RE = re.compile(
    r"^<!--\s*claude:\s*(?P<kind>always|archive|rule|skill)"
    r"(?:=(?P<name>[A-Za-z0-9_-]+))?\s*-->\s*$"
)
_HEADING_RE = re.compile(r"^(?P<hashes>#{2,6})\s+(?P<title>.*?)\s*$")
_FENCE_RE = re.compile(r"^\s*(```|~~~)")
_NAME_RE = re.compile(r"^[A-Za-z0-9_-]+$")


class SyncError(Exception):
    """Fatal problem with the markers, targets block, or output files."""


@dataclass(frozen=True)
class Target:
    kind: str                    # always | archive | rule | skill
    name: Optional[str] = None   # rule / skill name

    @property
    def key(self) -> str:
        return self.kind if self.name is None else f"{self.kind}={self.name}"


ALWAYS = Target("always")
ARCHIVE = Target("archive")


@dataclass
class Section:
    level: int
    title: str
    start: int                     # heading line index (0-based)
    own_end: int = 0               # exclusive; first child heading, or `end`
    end: int = 0                   # exclusive; next heading with level <= ours
    marker: Optional[Target] = None
    marker_line: Optional[int] = None
    parent: Optional["Section"] = None
    resolved: Target = ARCHIVE


@dataclass
class Targets:
    rules: Dict[str, dict] = field(default_factory=dict)    # name -> {when, paths}
    skills: Dict[str, dict] = field(default_factory=dict)   # name -> {description}


@dataclass
class Parsed:
    lines: List[str]
    targets: Targets
    sections: List[Section]
    warnings: List[str]
    hidden: Set[int]     # line indices never emitted (the targets block)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def _fence_mask(lines: List[str]) -> List[bool]:
    """True for every line inside (or delimiting) a fenced code block."""
    mask = [False] * len(lines)
    fence: Optional[str] = None
    for i, line in enumerate(lines):
        m = _FENCE_RE.match(line)
        if m and fence is None:
            fence = m.group(1)
            mask[i] = True
        elif m and m.group(1) == fence:
            fence = None
            mask[i] = True
        else:
            mask[i] = fence is not None
    return mask


def _validate_targets(data: object, lineno: int) -> Targets:
    where = f"{SOURCE_NAME}:{lineno}: claude-sync block"
    if not isinstance(data, dict):
        raise SyncError(f"{where} must be a YAML mapping with 'rules' / 'skills'")
    unknown = set(data) - {"rules", "skills"}
    if unknown:
        raise SyncError(f"{where} has unknown top-level key(s): {sorted(unknown)}")
    targets = Targets()
    for name, spec in (data.get("rules") or {}).items():
        name = str(name)
        if not _NAME_RE.match(name):
            raise SyncError(f"{where}: rule name {name!r} must match [A-Za-z0-9_-]+")
        spec = spec or {}
        if not isinstance(spec, dict) or not isinstance(spec.get("when"), str):
            raise SyncError(f"{where}: rule {name!r} needs a 'when:' string")
        paths = spec.get("paths") or []
        if not isinstance(paths, list) or not all(isinstance(p, str) for p in paths):
            raise SyncError(f"{where}: rule {name!r} 'paths:' must be a list of strings")
        extra = set(spec) - {"when", "paths"}
        if extra:
            raise SyncError(f"{where}: rule {name!r} has unknown key(s): {sorted(extra)}")
        targets.rules[name] = {"when": spec["when"].strip(),
                               "paths": [p.strip() for p in paths]}
    for name, spec in (data.get("skills") or {}).items():
        name = str(name)
        if not _NAME_RE.match(name):
            raise SyncError(f"{where}: skill name {name!r} must match [A-Za-z0-9_-]+")
        spec = spec or {}
        if not isinstance(spec, dict) or not isinstance(spec.get("description"), str):
            raise SyncError(f"{where}: skill {name!r} needs a 'description:' string")
        extra = set(spec) - {"description"}
        if extra:
            raise SyncError(f"{where}: skill {name!r} has unknown key(s): {sorted(extra)}")
        targets.skills[name] = {"description": spec["description"].strip()}
    return targets


def _parse_targets(lines: List[str], mask: List[bool]) -> Tuple[Targets, Tuple[int, int]]:
    for i, line in enumerate(lines):
        if mask[i] or not _TARGETS_OPEN_RE.match(line):
            continue
        for j in range(i + 1, len(lines)):
            if _COMMENT_CLOSE_RE.match(lines[j]):
                break
        else:
            raise SyncError(f"{SOURCE_NAME}:{i + 1}: 'claude-sync' block is never closed with '-->'")
        body = "\n".join(lines[i + 1:j])
        try:
            data = yaml.safe_load(body) or {}
        except yaml.YAMLError as e:
            raise SyncError(f"{SOURCE_NAME}:{i + 1}: claude-sync block is not valid YAML: {e}")
        return _validate_targets(data, i + 1), (i, j)
    raise SyncError(f"{SOURCE_NAME}: no '<!-- claude-sync' targets block found")


def _target_from_match(m: "re.Match[str]", targets: Targets, lineno: int) -> Target:
    kind, name = m.group("kind"), m.group("name")
    where = f"{SOURCE_NAME}:{lineno + 1}"
    if kind in ("always", "archive"):
        if name is not None:
            raise SyncError(f"{where}: '{kind}' marker takes no name")
        return Target(kind)
    if name is None:
        raise SyncError(f"{where}: '{kind}=' marker needs a name")
    known = targets.rules if kind == "rule" else targets.skills
    if name not in known:
        raise SyncError(
            f"{where}: unknown {kind} {name!r}; declare it in the claude-sync block "
            f"(known: {sorted(known) or 'none'})"
        )
    return Target(kind, name)


def _parse_sections(lines: List[str], mask: List[bool], targets: Targets) -> List[Section]:
    sections: List[Section] = []
    stack: List[Section] = []
    for i, line in enumerate(lines):
        if mask[i]:
            continue
        m = _HEADING_RE.match(line)
        if not m:
            continue
        level = len(m.group("hashes"))
        while stack and stack[-1].level >= level:
            stack.pop()
        sec = Section(level=level, title=m.group("title"), start=i,
                      parent=stack[-1] if stack else None)
        sections.append(sec)
        stack.append(sec)

    n = len(lines)
    for idx, sec in enumerate(sections):
        sec.end = n
        for nxt in sections[idx + 1:]:
            if nxt.level <= sec.level:
                sec.end = nxt.start
                break
        nxt_start = sections[idx + 1].start if idx + 1 < len(sections) else n
        sec.own_end = min(nxt_start, sec.end)

    consumed: Set[int] = set()
    for sec in sections:
        j = sec.start + 1
        while j < sec.own_end and lines[j].strip() == "":
            j += 1
        if j < sec.own_end and not mask[j]:
            m = _MARKER_RE.match(lines[j])
            if m:
                sec.marker = _target_from_match(m, targets, j)
                sec.marker_line = j
                consumed.add(j)

    for i, line in enumerate(lines):
        if not mask[i] and i not in consumed and _MARKER_RE.match(line):
            raise SyncError(
                f"{SOURCE_NAME}:{i + 1}: marker is not directly below a heading "
                f"(it would be silently ignored)"
            )
    return sections


def _resolve(sections: List[Section]) -> List[str]:
    warnings: List[str] = []
    for sec in sections:          # document order => parents resolve first
        if sec.marker is not None:
            sec.resolved = sec.marker
        elif sec.parent is not None:
            sec.resolved = sec.parent.resolved
        else:
            sec.resolved = ARCHIVE
            warnings.append(
                f"{SOURCE_NAME}:{sec.start + 1}: top-level section {sec.title!r} has no "
                f"<!-- claude: ... --> marker; treating as archive (Claude Code will not see it)"
            )
    return warnings


def parse(source: Path) -> Parsed:
    try:
        text = source.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise SyncError(f"{source} not found")
    lines = text.split("\n")
    mask = _fence_mask(lines)
    targets, (b0, b1) = _parse_targets(lines, mask)
    sections = _parse_sections(lines, mask, targets)
    warnings = _resolve(sections)
    return Parsed(lines=lines, targets=targets, sections=sections,
                  warnings=warnings, hidden=set(range(b0, b1 + 1)))


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _own_text(parsed: Parsed, sec: Section) -> List[str]:
    return [
        parsed.lines[i]
        for i in range(sec.start, sec.own_end)
        if i != sec.marker_line and i not in parsed.hidden
    ]


def _join(blocks: List[List[str]]) -> str:
    out: List[str] = []
    for block in blocks:
        while block and block[-1].strip() == "":
            block = block[:-1]
        if not block:
            continue
        out.extend(block)
        out.append("")
    return "\n".join(out).rstrip("\n") + "\n" if out else ""


def _frontmatter(data: dict) -> str:
    body = yaml.safe_dump(data, sort_keys=False, allow_unicode=True, width=10_000)
    return f"---\n{body}---\n"


def _archive_roots(parsed: Parsed) -> List[Section]:
    return [
        s for s in parsed.sections
        if s.resolved == ARCHIVE and (s.parent is None or s.parent.resolved != ARCHIVE)
    ]


def _render_claude_md(parsed: Parsed, body: str) -> str:
    t = parsed.targets
    parts = [
        BANNER,
        "# Agent Collaboration Guide (Claude Code subset)",
        "",
        f"This file is generated from [`{SOURCE_NAME}`]({SOURCE_NAME}), the canonical "
        "collaboration guide for this repo (shared with Cursor).  It carries the always-on "
        "sections; topic guidance lives in `.claude/rules/` (rule map at the end) and loads "
        "automatically when you read or edit matching files.  **Do not edit this file**: edit "
        f"`{SOURCE_NAME}`, then run `{SYNC_CMD}`.  If the two ever disagree, `{SOURCE_NAME}` wins.",
        "",
        body.rstrip("\n"),
        "",
        "## Rule map (generated)",
        "",
        "Each rule below loads automatically when you open a file matching its `paths` "
        "frontmatter with the **Read tool** (shell reads such as `cat` or `sed` via Bash do not "
        "trigger it), and it is restored after context compaction.  **If you start work of one "
        "of these kinds without having Read a matching file, Read the rule file first.**",
        "",
    ]
    scoped = [(n, s) for n, s in t.rules.items() if s["paths"]]
    unscoped = [n for n, s in t.rules.items() if not s["paths"]]
    for name, spec in scoped:
        parts.append(f"- **When {spec['when']}:** [`{RULES_DIR}/{name}.md`]({RULES_DIR}/{name}.md)")
    if unscoped:
        parts.append("")
        parts.append("Always loaded (no path filter): " + ", ".join(
            f"[`{RULES_DIR}/{n}.md`]({RULES_DIR}/{n}.md)" for n in unscoped))
    if t.skills:
        parts += ["", "## Skills (generated)", ""]
        for name, spec in t.skills.items():
            parts.append(f"- `/{name}`: {spec['description']} "
                         f"([`{SKILLS_DIR}/{name}/SKILL.md`]({SKILLS_DIR}/{name}/SKILL.md))")
    roots = _archive_roots(parsed)
    if roots:
        parts += [
            "", "## Archive index (generated)", "",
            f"These sections exist only in `{SOURCE_NAME}`; open them there when the topic comes up.",
            "",
        ]
        parts += [f"- {s.title}" for s in roots]
    return "\n".join(parts).rstrip("\n") + "\n"


def _render_rule(name: str, spec: dict, blocks: List[List[str]]) -> str:
    head = _frontmatter({"paths": spec["paths"]}) if spec["paths"] else ""
    scope = ("Loads automatically for files matching the `paths` above" if spec["paths"]
             else "Always loaded (no path filter)")
    intro = (
        f"# Rule: {name}\n\n"
        f"**When:** {spec['when']}.  {scope}.  Source: the sections of "
        f"[`{SOURCE_NAME}`]({SOURCE_NAME}) marked `rule={name}`; edit there, then run "
        f"`{SYNC_CMD}`.\n\n"
    )
    return head + BANNER + "\n" + intro + _join(blocks)


def _render_skill(name: str, spec: dict, blocks: List[List[str]]) -> str:
    head = _frontmatter({"name": name, "description": spec["description"]})
    return head + BANNER + "\n" + _join(blocks)


def generate(parsed: Parsed) -> Tuple[Dict[str, str], List[str]]:
    """Return ``{repo-relative path: content}`` plus warnings."""
    warnings = list(parsed.warnings)
    by_target: Dict[str, List[List[str]]] = defaultdict(list)
    for sec in parsed.sections:
        by_target[sec.resolved.key].append(_own_text(parsed, sec))

    outputs: Dict[str, str] = {
        CLAUDE_MD: _render_claude_md(parsed, _join(by_target.get("always", []))),
    }
    for name, spec in parsed.targets.rules.items():
        blocks = by_target.get(f"rule={name}")
        if not blocks:
            warnings.append(f"rule {name!r} is declared but no section is marked rule={name}; "
                            f"no file written")
            continue
        outputs[f"{RULES_DIR}/{name}.md"] = _render_rule(name, spec, blocks)
    for name, spec in parsed.targets.skills.items():
        blocks = by_target.get(f"skill={name}")
        if not blocks:
            warnings.append(f"skill {name!r} is declared but no section is marked skill={name}; "
                            f"no file written")
            continue
        outputs[f"{SKILLS_DIR}/{name}/SKILL.md"] = _render_skill(name, spec, blocks)
    return outputs, warnings


# ---------------------------------------------------------------------------
# Filesystem
# ---------------------------------------------------------------------------

def _diff(repo: Path, outputs: Dict[str, str]) -> Tuple[List[str], List[str], List[str]]:
    """Return (stale_or_missing, orphans, conflicts) as repo-relative paths."""
    stale: List[str] = []
    conflicts: List[str] = []
    for rel, content in outputs.items():
        path = repo / rel
        if not path.exists():
            stale.append(rel)
            continue
        existing = path.read_text(encoding="utf-8")
        if existing == content:
            continue
        stale.append(rel)
        if BANNER not in existing:
            conflicts.append(rel)

    candidates = list((repo / RULES_DIR).glob("*.md")) + list((repo / SKILLS_DIR).glob("*/SKILL.md"))
    orphans = [
        p.relative_to(repo).as_posix() for p in sorted(candidates)
        if p.relative_to(repo).as_posix() not in outputs
        and BANNER in p.read_text(encoding="utf-8")
    ]
    return stale, orphans, conflicts


def _remove(repo: Path, rel: str) -> None:
    path = repo / rel
    path.unlink()
    parent = path.parent
    if parent != repo / RULES_DIR and parent.is_dir() and not any(parent.iterdir()):
        parent.rmdir()


def main(argv: Optional[List[str]] = None, *, repo: Optional[Path] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--check", action="store_true",
                    help="report stale / missing / orphaned generated files and exit 1 "
                         "instead of writing anything")
    ap.add_argument("--quiet", action="store_true",
                    help="print only when something changed (session-start hook mode)")
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing hand-written (banner-less) output file")
    args = ap.parse_args(argv)
    repo = repo or _REPO_ROOT

    try:
        parsed = parse(repo / SOURCE_NAME)
        outputs, warnings = generate(parsed)
        stale, orphans, conflicts = _diff(repo, outputs)
        if conflicts and not args.force:
            raise SyncError(
                "refusing to overwrite hand-written file(s) without the generated banner "
                f"(rerun with --force to replace): {', '.join(conflicts)}"
            )
    except SyncError as e:
        print(f"sync_agent_notes: error: {e}", file=sys.stderr)
        return 2

    for w in warnings:
        print(f"sync_agent_notes: warning: {w}", file=sys.stderr)

    if args.check:
        for rel in stale:
            print(f"stale: {rel}", file=sys.stderr)
        for rel in orphans:
            print(f"orphan: {rel}", file=sys.stderr)
        if stale or orphans:
            print(f"sync_agent_notes: {len(stale)} stale, {len(orphans)} orphan; run `{SYNC_CMD}`",
                  file=sys.stderr)
            return 1
        if not args.quiet:
            print("sync_agent_notes: up to date")
        return 0

    for rel in stale:
        atomic_write_text(outputs[rel], repo / rel)
    for rel in orphans:
        _remove(repo, rel)
    if stale or orphans:
        bits = []
        if stale:
            bits.append(f"wrote {len(stale)} file(s): {', '.join(stale)}")
        if orphans:
            bits.append(f"removed {len(orphans)} orphan(s): {', '.join(orphans)}")
        print("sync_agent_notes: " + "; ".join(bits))
    elif not args.quiet:
        print("sync_agent_notes: up to date")
    return 0


if __name__ == "__main__":
    sys.exit(main())
