#!/usr/bin/env python3
"""PreToolUse hook: ask before any tool call that names a path under the home
directory outside the project boundary (AGENT_NOTES / CLAUDE.md § "File Access
Boundary").  Enforcement, not documentation: on 2026-09-17 five planning
subagents listed caches under ~ against the written rule and their brief, and
only three reported it.

Reads the hook JSON on stdin.  For Bash it scans the command text for
home-directory paths (``/Users/<user>/...``, ``~/...``, ``$HOME/...``) and for
tokens that climb out of the working directory (``../``).  For the file tools
(Read, Edit, Write, NotebookEdit, Glob, Grep) it resolves the path fields
against the call's cwd.  Any hit outside the allowlist prints a
``permissionDecision: "ask"`` so Roger sees the command and decides; anything
else prints nothing (the normal permission flow applies).

Allowlist (all prefixes):
  - the repository itself
  - ~/.claude/            (Claude Code's own state: memory, plans, settings)
  - /tmp and /private/tmp   (scratch generally, including the claude-* session scratchpads)
  - ~/.cursor/projects/*/terminals/ and */agent-tools/   (Cursor tool infrastructure)

Known false positive: the Bash scan sees text, not shell syntax, so a heredoc
whose *body* mentions a home path (for example an edit to the boundary rule
itself) also triggers the ask.  Since 2026-09-25 the reason says when every
match is inside a heredoc body, so a prose edit can be judged at a glance; it
stays an ask because a heredoc-fed script can open the path just as well as
mention it.  Prose that quotes home paths is better written with the Write /
Edit tools, which the hook checks by path only.
"""
import json
import os
import re
import sys

HOME = os.path.expanduser("~")
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ALLOWED_PREFIXES = (
    REPO,
    os.path.join(HOME, ".claude"),
    "/tmp",  # Roger, 2026-09-25: /tmp generally, not only the claude-* scratchpads
    "/private/tmp",
)
ALLOWED_PATTERNS = (
    re.compile(re.escape(os.path.join(HOME, ".cursor", "projects")) + r"/[^/]+/(terminals|agent-tools)(/|$)"),
)
FILE_TOOLS = {"Read", "Edit", "Write", "NotebookEdit", "Glob", "Grep"}
HOME_PATH = re.compile(r"(?:" + re.escape(HOME) + r"|~|\$HOME|\$\{HOME\})(?:/[^\s\"'`;|&)>]*)?")
CLIMB = re.compile(r"(?<![\w./])\.\./[^\s\"'`;|&)>]*|(?<![\w./])\.\.(?=[\s;|&)]|$)")
# ``<<EOF``, ``<<-EOF``, ``<<'EOF'``, ``<<"EOF"``, ``<<\EOF``; not the here-string ``<<<``
# a bare delimiter must start with a letter or underscore and may not be followed by ``)``,
# so neither ``$((1 << 3))`` nor ``$((x << y))`` is a marker
HEREDOC = re.compile(r"(?<!<)<<-?[ \t]*(?:'([^'\n]+)'|\"([^\"\n]+)\"|\\?([A-Za-z_][\w.-]*))(?![<)])")


def normalize(p: str) -> str:
    p = p.replace("${HOME}", HOME).replace("$HOME", HOME)
    if p == "~" or p.startswith("~/"):
        p = HOME + p[1:]
    return os.path.normpath(p)


def allowed(p: str) -> bool:
    p = os.path.normpath(p)
    if any(p == a.rstrip("/") or p.startswith(a if a.endswith("-") else a.rstrip("/") + "/") for a in ALLOWED_PREFIXES):
        return True
    return any(pat.match(p) for pat in ALLOWED_PATTERNS)


def heredoc_bodies(text: str) -> list:
    """Spans (start, end) of heredoc bodies in a shell command.

    A marker is ``<<``, ``<<-`` and a delimiter (bare, quoted or
    backslashed); the body runs from the next line to the line holding the
    delimiter alone (leading tabs allowed), or to the end of the text when it
    is unterminated.  Markers found inside a body are skipped, so a ``<<`` in
    a Python script does not start a nested body.  Only the ask *reason* uses
    this; it never decides whether to ask.
    """
    spans = []
    pos = 0  # a body on the same line as an earlier marker starts after that body
    for m in HEREDOC.finditer(text):
        if any(s <= m.start() < e for s, e in spans):
            continue
        delim = m.group(1) or m.group(2) or m.group(3)
        line_end = text.find("\n", m.end())
        if line_end == -1:
            break
        start = max(line_end + 1, pos)
        term = re.compile(r"^\t*" + re.escape(delim) + r"[ \t]*$", re.M).search(text, start)
        end = term.start() if term else len(text)
        spans.append((start, end))
        pos = term.end() + 1 if term else len(text)
    return spans


def offending_in_text(text: str, cwd: str) -> tuple:
    """Return (sorted offending paths, all_in_heredoc).

    ``all_in_heredoc`` is True when every occurrence of every offending path
    lies inside a heredoc body, which usually means prose or script text
    rather than a shell argument.
    """
    bodies = heredoc_bodies(text)
    hits: dict = {}  # path -> every occurrence so far inside a heredoc body

    def add(p: str, at: int) -> None:
        inside = any(s <= at < e for s, e in bodies)
        hits[p] = hits.get(p, True) and inside

    for m in HOME_PATH.finditer(text):
        p = normalize(m.group(0))
        if p.startswith(HOME) and not allowed(p):
            add(p, m.start())
    for m in CLIMB.finditer(text):
        p = os.path.normpath(os.path.join(cwd, m.group(0)))
        if not allowed(p):
            add(p, m.start())
    return sorted(hits), bool(hits) and all(hits.values())


def main() -> int:
    try:
        data = json.load(sys.stdin)
    except Exception:
        return 0
    tool = data.get("tool_name", "")
    inp = data.get("tool_input") or {}
    cwd = data.get("cwd") or os.getcwd()
    hits: list = []
    in_heredoc = False
    if tool == "Bash":
        hits, in_heredoc = offending_in_text(inp.get("command", "") or "", cwd)
    elif tool in FILE_TOOLS:
        for key in ("file_path", "path", "notebook_path"):
            v = inp.get(key)
            if not v:
                continue
            p = normalize(v)
            if not os.path.isabs(p):
                p = os.path.normpath(os.path.join(cwd, p))
            if p.startswith(HOME) and not allowed(p):
                hits.append(p)
        # Glob/Grep patterns can carry absolute paths too
        pat = inp.get("pattern")
        if isinstance(pat, str) and (pat.startswith(HOME) or pat.startswith("~")):
            p = normalize(pat.split("*")[0])
            if not allowed(p):
                hits.append(p)
    if not hits:
        return 0
    shown = ", ".join(hits[:4]) + (" ..." if len(hits) > 4 else "")
    reason = f"File-access boundary: {tool} names a path outside the project ({shown}). "
    if in_heredoc:
        reason += (
            "Every match is inside a heredoc body, so this is probably prose or script "
            "text rather than a shell argument; check whether the body only mentions the "
            "path or opens it. "
        )
    reason += "The rule (CLAUDE.md) says ask Roger first; approve only if intended."
    out = {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "ask",
            "permissionDecisionReason": reason,
        }
    }
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
