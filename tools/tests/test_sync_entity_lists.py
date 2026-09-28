"""Tests for tools/sync_entity_lists.py.

The lists are derived views of the instruction directories; these tests
pin the derivation (which files count, ordering, verbatim descriptions),
the ``--check`` gate, and -- via ``test_repo_lists_are_in_sync`` -- that
the checked-in lists never lag the real corpus again.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# Make ``tools.sync_entity_lists`` importable without an editable install.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools import sync_entity_lists as sel  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _write(path: Path, doc) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    d = tmp_path / "data"
    # Roles: deliberately written out of alphabetical order.
    _write(d / "roles/instructions/zebra_keeper.json",
           {"description": "Looks after zebras.", "instruction": [{"pos": "You keep zebras."}]})
    _write(d / "roles/instructions/assistant.json", {"description": "A general-purpose aide."})
    # The bare corpus default has no description at all.
    _write(d / "roles/instructions/default.json", {"instruction": [{"pos": ""}]})
    (d / "roles/instructions/ROLES_TO_ADD.md").write_text("# not json\n", encoding="utf-8")
    # Traits, including a blank description and a non-ASCII one.
    _write(d / "traits/instructions/blunt.json",
           {"positive_label": "blunt", "negative_label": "tactful",
            "description": "This means saying it straight."})
    _write(d / "traits/instructions/calm.json", {"positive_label": "calm", "description": "   "})
    _write(d / "traits/instructions/analytical.json",
           {"positive_label": "analytical",
            "description": "This means breaking things down -- étape by étape."})
    return d


# ---------------------------------------------------------------------------
# build_list / render / describe_drift
# ---------------------------------------------------------------------------

class TestBuildList:
    def test_roles_sorted_and_skips_missing_description(self, data_dir: Path) -> None:
        entries, skipped = sel.build_list(data_dir, "roles")
        assert list(entries) == ["assistant", "zebra_keeper"]
        assert entries["zebra_keeper"] == "Looks after zebras."
        assert skipped == ["default"]
        assert "ROLES_TO_ADD" not in entries  # non-JSON files are ignored

    def test_traits_skip_blank_description_and_keep_text_verbatim(self, data_dir: Path) -> None:
        entries, skipped = sel.build_list(data_dir, "traits")
        assert list(entries) == ["analytical", "blunt"]
        assert entries["analytical"] == "This means breaking things down -- étape by étape."
        assert skipped == ["calm"]

    def test_missing_directory_raises(self, tmp_path: Path) -> None:
        with pytest.raises(sel.SyncError, match="not a directory"):
            sel.build_list(tmp_path, "roles")

    def test_invalid_json_raises_with_path(self, data_dir: Path) -> None:
        (data_dir / "traits/instructions/bad.json").write_text("{not json", encoding="utf-8")
        with pytest.raises(sel.SyncError, match="bad.json"):
            sel.build_list(data_dir, "traits")


class TestRender:
    def test_sorted_unescaped_unicode_trailing_newline(self) -> None:
        text = sel.render({"b": "x", "a": "étape"})
        assert text.endswith("\n")
        assert "étape" in text          # ensure_ascii=False
        assert text.index('"a"') < text.index('"b"')
        assert json.loads(text) == {"a": "étape", "b": "x"}


class TestDescribeDrift:
    def test_missing(self) -> None:
        assert sel.describe_drift(None, {"a": "x"}) == "missing"

    def test_not_json(self) -> None:
        assert sel.describe_drift("{oops", {"a": "x"}) == "not valid JSON"

    def test_counts(self) -> None:
        old = json.dumps({"keep": "same", "gone": "old", "edit": "before"})
        new = {"keep": "same", "edit": "after", "new": "added"}
        assert sel.describe_drift(old, new) == "1 added, 1 removed, 1 changed"

    def test_formatting_only(self) -> None:
        old = json.dumps({"a": "x"}, indent=4)  # same content, different layout
        assert sel.describe_drift(old, {"a": "x"}) == "0 added, 0 removed, 0 changed, formatting only"


# ---------------------------------------------------------------------------
# main / CLI behaviour
# ---------------------------------------------------------------------------

class TestMain:
    def test_regenerate_writes_both_lists(self, data_dir: Path, capsys: pytest.CaptureFixture) -> None:
        assert sel.main(["--data_dir", str(data_dir)]) == 0
        roles = json.loads((data_dir / "roles/role_list.json").read_text(encoding="utf-8"))
        traits = json.loads((data_dir / "traits/trait_list.json").read_text(encoding="utf-8"))
        assert roles == {"assistant": "A general-purpose aide.", "zebra_keeper": "Looks after zebras."}
        assert traits == {
            "analytical": "This means breaking things down -- étape by étape.",
            "blunt": "This means saying it straight.",
        }
        out = capsys.readouterr().out
        assert "rewrote (2 entries; missing)" in out
        assert "skipped (no description): default" in out
        assert "skipped (no description): calm" in out

    def test_check_never_writes_and_tracks_state(self, data_dir: Path, capsys: pytest.CaptureFixture) -> None:
        # Missing lists -> stale, nothing written.
        assert sel.main(["--check", "--data_dir", str(data_dir)]) == 1
        assert "STALE (missing)" in capsys.readouterr().out
        assert not (data_dir / "roles/role_list.json").exists()
        # Regenerate -> clean.
        assert sel.main(["--data_dir", str(data_dir)]) == 0
        assert sel.main(["--check", "--data_dir", str(data_dir)]) == 0
        assert "up to date (2 entries)" in capsys.readouterr().out
        # A new trait file makes the trait list stale (roles still clean).
        _write(data_dir / "traits/instructions/zany.json", {"description": "This means being zany."})
        assert sel.main(["--check", "--data_dir", str(data_dir)]) == 1
        out = capsys.readouterr().out
        assert "trait_list.json: STALE (1 added, 0 removed, 0 changed)" in out
        assert "role_list.json: up to date" in out
        # Check must not have written the new entry.
        assert "zany" not in (data_dir / "traits/trait_list.json").read_text(encoding="utf-8")

    def test_hand_edit_is_overwritten(self, data_dir: Path) -> None:
        assert sel.main(["--data_dir", str(data_dir)]) == 0
        p = data_dir / "traits/trait_list.json"
        doc = json.loads(p.read_text(encoding="utf-8"))
        doc["blunt"] = "hand edit"
        p.write_text(json.dumps(doc), encoding="utf-8")
        assert sel.main(["--check", "--data_dir", str(data_dir)]) == 1
        assert sel.main(["--data_dir", str(data_dir)]) == 0
        assert json.loads(p.read_text(encoding="utf-8"))["blunt"] == "This means saying it straight."

    def test_kinds_subset_touches_only_that_list(self, data_dir: Path) -> None:
        assert sel.main(["--data_dir", str(data_dir), "--kinds", "traits"]) == 0
        assert (data_dir / "traits/trait_list.json").exists()
        assert not (data_dir / "roles/role_list.json").exists()

    def test_invalid_json_exits_2(self, data_dir: Path, capsys: pytest.CaptureFixture) -> None:
        (data_dir / "roles/instructions/bad.json").write_text("{not json", encoding="utf-8")
        assert sel.main(["--data_dir", str(data_dir)]) == 2
        assert "bad.json" in capsys.readouterr().err

    def test_env_var_selects_data_dir(self, data_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ASSISTANT_AXIS_DATA_DIR", str(data_dir))
        assert sel.main([]) == 0
        assert (data_dir / "roles/role_list.json").exists()

    def test_quiet_is_silent_when_clean(self, data_dir: Path, capsys: pytest.CaptureFixture) -> None:
        assert sel.main(["--data_dir", str(data_dir)]) == 0
        capsys.readouterr()
        assert sel.main(["--check", "--quiet", "--data_dir", str(data_dir)]) == 0
        assert capsys.readouterr().out == ""


# ---------------------------------------------------------------------------
# Guard against the lists lagging the real corpus again
# ---------------------------------------------------------------------------

def test_repo_lists_are_in_sync() -> None:
    """The checked-in lists must match ``data/*/instructions/``.

    If this fails you added or edited an instruction file without
    regenerating the lists: run ``uv run python tools/sync_entity_lists.py``.
    """
    repo_data = _REPO_ROOT / "data"
    if not (repo_data / "traits" / "instructions").is_dir():
        pytest.skip("repo data directory not present")
    assert sel.main(["--check", "--quiet", "--data_dir", str(repo_data)]) == 0
