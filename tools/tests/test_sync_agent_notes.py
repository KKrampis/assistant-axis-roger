"""Tests for tools/sync_agent_notes.py."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools import sync_agent_notes as sam  # noqa: E402


NOTES = """# Guide

Intro text before any section.

<!-- claude-sync
rules:
  alpha:
    when: doing alpha things
    paths:
      - "src/alpha/**/*.py"
  beta:
    when: every session
skills:
  sk:
    description: A skill for testing
-->

## A
<!-- claude: always -->

A body.

### A1

A1 body (inherits always).

### A2
<!-- claude: rule=alpha -->

A2 body.

#### A2a

A2a body (inherits alpha).

#### A2b
<!-- claude: archive -->

A2b body (archive).

## B
<!-- claude: rule=beta -->

B body.

```text
## Not a heading
<!-- claude: rule=nope -->
```

## C
<!-- claude: skill=sk -->

C body.

## D

D body (unmarked, so archive).
"""


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    # atomic_write_text stages under TMPDIR; keep that inside the test dir.
    monkeypatch.setenv("TMPDIR", str(tmp_path / "tmp"))
    r = tmp_path / "repo"
    r.mkdir()
    (r / "AGENT_NOTES.md").write_text(NOTES, encoding="utf-8")
    return r


def _resolved(repo: Path) -> dict:
    parsed = sam.parse(repo / "AGENT_NOTES.md")
    return {s.title: s.resolved.key for s in parsed.sections}, parsed


# ---------------------------------------------------------------------------


def test_marker_resolution_and_inheritance(repo: Path) -> None:
    resolved, parsed = _resolved(repo)
    assert resolved == {
        "A": "always",
        "A1": "always",
        "A2": "rule=alpha",
        "A2a": "rule=alpha",
        "A2b": "archive",
        "B": "rule=beta",
        "C": "skill=sk",
        "D": "archive",
    }
    assert "Not a heading" not in resolved            # fenced heading ignored
    assert len(parsed.warnings) == 1 and "'D'" in parsed.warnings[0]


def test_generate_outputs(repo: Path) -> None:
    _, parsed = _resolved(repo)
    outputs, warnings = sam.generate(parsed)
    assert set(outputs) == {
        "CLAUDE.md",
        ".claude/rules/alpha.md",
        ".claude/rules/beta.md",
        ".claude/skills/sk/SKILL.md",
    }
    for content in outputs.values():
        assert sam.BANNER in content
        for marker in ("always", "rule=alpha", "rule=beta", "skill=sk", "archive"):
            assert f"<!-- claude: {marker} -->" not in content   # markers stripped
        assert "claude-sync" not in content            # targets block never emitted

    claude = outputs["CLAUDE.md"]
    assert "A body." in claude and "A1 body" in claude
    assert "A2 body" not in claude and "B body" not in claude
    assert "**When doing alpha things:**" in claude and ".claude/rules/alpha.md" in claude
    assert "Always loaded (no path filter)" in claude and ".claude/rules/beta.md" in claude
    assert "`/sk`: A skill for testing" in claude
    assert "## Archive index (generated)" in claude
    assert "- A2b" in claude and "- D" in claude and "- A2a" not in claude

    alpha = outputs[".claude/rules/alpha.md"]
    assert alpha.startswith("---\npaths:\n- src/alpha/**/*.py\n---\n" + sam.BANNER)
    assert "A2 body" in alpha and "A2a body" in alpha and "A2b body" not in alpha

    beta = outputs[".claude/rules/beta.md"]
    assert beta.startswith(sam.BANNER)                 # no frontmatter without paths
    assert "B body" in beta and "## Not a heading" in beta   # fenced text copied verbatim

    skill = outputs[".claude/skills/sk/SKILL.md"]
    assert skill.startswith("---\nname: sk\ndescription: A skill for testing\n---\n" + sam.BANNER)
    assert "C body" in skill
    assert warnings and "'D'" in warnings[0]


def test_main_writes_then_is_idempotent(repo: Path, capsys: pytest.CaptureFixture) -> None:
    assert sam.main([], repo=repo) == 0
    first = {p: (repo / p).read_text() for p in
             ["CLAUDE.md", ".claude/rules/alpha.md", ".claude/rules/beta.md",
              ".claude/skills/sk/SKILL.md"]}
    assert "wrote 4 file(s)" in capsys.readouterr().out

    assert sam.main(["--quiet"], repo=repo) == 0
    assert capsys.readouterr().out == ""               # quiet: silent when unchanged
    assert {p: (repo / p).read_text() for p in first} == first
    assert sam.main(["--check"], repo=repo) == 0


def test_check_detects_drift_and_missing(repo: Path, capsys: pytest.CaptureFixture) -> None:
    assert sam.main([], repo=repo) == 0
    with (repo / "CLAUDE.md").open("a") as fh:
        fh.write("\nlocal edit\n")
    assert sam.main(["--check"], repo=repo) == 1
    assert "stale: CLAUDE.md" in capsys.readouterr().err

    assert sam.main([], repo=repo) == 0                # regenerated (banner present)
    assert "local edit" not in (repo / "CLAUDE.md").read_text()
    assert sam.main(["--check"], repo=repo) == 0

    (repo / ".claude/rules/alpha.md").unlink()
    assert sam.main(["--check"], repo=repo) == 1
    assert "stale: .claude/rules/alpha.md" in capsys.readouterr().err


def test_source_edit_propagates(repo: Path) -> None:
    assert sam.main([], repo=repo) == 0
    notes = repo / "AGENT_NOTES.md"
    notes.write_text(notes.read_text().replace("A body.", "A body, revised."))
    assert sam.main(["--check"], repo=repo) == 1
    assert sam.main([], repo=repo) == 0
    assert "A body, revised." in (repo / "CLAUDE.md").read_text()


def test_orphaned_generated_file_removed_but_handwritten_kept(repo: Path) -> None:
    assert sam.main([], repo=repo) == 0
    old = repo / ".claude/rules/old.md"
    old.write_text(sam.BANNER + "\nstale rule\n")
    mine = repo / ".claude/rules/mine.md"
    mine.write_text("# my own rule, not generated\n")
    (repo / ".claude/skills/gone").mkdir()
    (repo / ".claude/skills/gone/SKILL.md").write_text(sam.BANNER + "\nstale skill\n")

    assert sam.main(["--check"], repo=repo) == 1
    assert sam.main([], repo=repo) == 0
    assert not old.exists()
    assert not (repo / ".claude/skills/gone").exists()   # empty skill dir pruned
    assert mine.exists()


def test_refuses_to_clobber_handwritten_claude_md(repo: Path) -> None:
    (repo / "CLAUDE.md").write_text("# hand-written\n")
    assert sam.main([], repo=repo) == 2
    assert (repo / "CLAUDE.md").read_text() == "# hand-written\n"
    assert sam.main(["--force"], repo=repo) == 0
    assert sam.BANNER in (repo / "CLAUDE.md").read_text()


def test_unknown_rule_name_is_error(repo: Path, capsys: pytest.CaptureFixture) -> None:
    notes = repo / "AGENT_NOTES.md"
    notes.write_text(notes.read_text().replace("rule=alpha", "rule=nope"))
    assert sam.main([], repo=repo) == 2
    assert "unknown rule 'nope'" in capsys.readouterr().err
    assert not (repo / "CLAUDE.md").exists()


def test_stray_marker_is_error(repo: Path, capsys: pytest.CaptureFixture) -> None:
    notes = repo / "AGENT_NOTES.md"
    notes.write_text(notes.read_text().replace(
        "A1 body (inherits always).", "A1 body.\n<!-- claude: always -->"))
    assert sam.main([], repo=repo) == 2
    assert "not directly below a heading" in capsys.readouterr().err


def test_missing_targets_block_is_error(repo: Path) -> None:
    notes = repo / "AGENT_NOTES.md"
    text = notes.read_text()
    start, end = text.index("<!-- claude-sync"), text.index("-->") + len("-->")
    notes.write_text(text[:start] + text[end:])
    assert sam.main([], repo=repo) == 2


def test_declared_but_unused_rule_warns(repo: Path, capsys: pytest.CaptureFixture) -> None:
    notes = repo / "AGENT_NOTES.md"
    notes.write_text(notes.read_text().replace(
        "  beta:\n    when: every session\n",
        "  beta:\n    when: every session\n  gamma:\n    when: never used\n"))
    assert sam.main([], repo=repo) == 0
    assert "rule 'gamma' is declared but no section" in capsys.readouterr().err
    assert not (repo / ".claude/rules/gamma.md").exists()
