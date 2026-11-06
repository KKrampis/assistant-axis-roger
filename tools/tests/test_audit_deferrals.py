"""Tests for tools.audit_deferrals (Phase 6d-2 / category invariants)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from assistant_axis import deferral_registry as deferral  # noqa: E402

# audit_deferrals lives in tools/, which is not packaged.  Import via
# importlib so we don't need to reorganize.
import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "audit_deferrals",
    _REPO_ROOT / "tools" / "audit_deferrals.py",
)
audit_deferrals = importlib.util.module_from_spec(_spec)
sys.modules["audit_deferrals"] = audit_deferrals
_spec.loader.exec_module(audit_deferrals)  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _stub_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "fake_repo"
    repo.mkdir()
    (repo / ".git").mkdir()  # so deferral_registry recognises it
    return repo


def _add_deferral(repo: Path, **kwargs) -> deferral.DeferralEntry:
    """Append one entry to the test repo's registry."""
    return deferral.append_deferral(repo_root=repo, **kwargs)


def _patch_git_ls_files(monkeypatch, files: set[str]) -> None:
    """Stub out the git subprocess call so tests don't depend on the
    real git tree."""
    monkeypatch.setattr(audit_deferrals, "_git_ls_files", lambda root: files)


# ---------------------------------------------------------------------------
# Roll-up
# ---------------------------------------------------------------------------

def test_audit_empty_registry(tmp_path: Path, monkeypatch) -> None:
    repo = _stub_repo(tmp_path)
    _patch_git_ls_files(monkeypatch, set())
    report = audit_deferrals.audit(repo_root=repo)
    assert report.issues == []
    assert report.counts_by_category == {}


def test_audit_counts_categories(tmp_path: Path, monkeypatch) -> None:
    repo = _stub_repo(tmp_path)
    _add_deferral(repo, path_glob="a.json", reason="r",
                  category=deferral.DeferralCategory.LEGACY_BARE)
    _add_deferral(repo, path_glob="b.json", reason="r",
                  category=deferral.DeferralCategory.OPERATIONAL)
    _add_deferral(repo, path_glob="c.json", reason="r",
                  category=deferral.DeferralCategory.OPERATIONAL)
    _patch_git_ls_files(monkeypatch, set())
    report = audit_deferrals.audit(repo_root=repo)
    assert report.counts_by_category[deferral.DeferralCategory.LEGACY_BARE] == 1
    assert report.counts_by_category[deferral.DeferralCategory.OPERATIONAL] == 2


# ---------------------------------------------------------------------------
# orphan_no_producer: promotion alarm
# ---------------------------------------------------------------------------

def test_orphan_promoted_raises_error(tmp_path: Path, monkeypatch) -> None:
    """If the named producer_script reappears in git ls-files, the
    orphan was promoted -- the maintainer should lift the deferral."""
    repo = _stub_repo(tmp_path)
    _add_deferral(
        repo,
        path_glob="roger/canonical_angles_*.png", reason="orphan PNG",
        category=deferral.DeferralCategory.ORPHAN_NO_PRODUCER,
        producer_script="results_analysis/canonical_angles.py",
    )
    _patch_git_ls_files(
        monkeypatch, {"results_analysis/canonical_angles.py"}
    )
    report = audit_deferrals.audit(repo_root=repo)
    assert len(report.errors()) == 1
    assert report.errors()[0].invariant == "orphan_promoted"


def test_orphan_with_no_producer_script_no_error(
    tmp_path: Path, monkeypatch,
) -> None:
    """An orphan entry without producer_script can't trip the
    promotion alarm; backfilled v1 entries fall in this bucket and
    must remain quiet."""
    repo = _stub_repo(tmp_path)
    _add_deferral(
        repo,
        path_glob="roger/canonical_angles_*.png", reason="orphan PNG",
        category=deferral.DeferralCategory.ORPHAN_NO_PRODUCER,
    )
    _patch_git_ls_files(
        monkeypatch, {"results_analysis/canonical_angles.py"}
    )
    report = audit_deferrals.audit(repo_root=repo)
    assert report.errors() == []
    assert report.warnings() == []


def test_orphan_producer_still_orphan_no_error(
    tmp_path: Path, monkeypatch,
) -> None:
    """If the named producer_script is NOT in git, no alarm."""
    repo = _stub_repo(tmp_path)
    _add_deferral(
        repo,
        path_glob="roger/canonical_angles_*.png", reason="orphan PNG",
        category=deferral.DeferralCategory.ORPHAN_NO_PRODUCER,
        producer_script="/tmp/_canonical_angles_v1.py",
    )
    _patch_git_ls_files(monkeypatch, {"results_analysis/something_else.py"})
    report = audit_deferrals.audit(repo_root=repo)
    assert report.errors() == []


# ---------------------------------------------------------------------------
# superseded: replaced_by hygiene
# ---------------------------------------------------------------------------

def test_superseded_missing_replaced_by_warns(
    tmp_path: Path, monkeypatch,
) -> None:
    repo = _stub_repo(tmp_path)
    _add_deferral(
        repo, path_glob="roger/old.json", reason="legacy",
        category=deferral.DeferralCategory.SUPERSEDED,
    )
    _patch_git_ls_files(monkeypatch, set())
    report = audit_deferrals.audit(repo_root=repo)
    assert len(report.warnings()) == 1
    assert report.warnings()[0].invariant == "superseded_missing_replaced_by"
    assert report.errors() == []


def test_superseded_replaced_by_missing_errors(
    tmp_path: Path, monkeypatch,
) -> None:
    """``replaced_by`` glob matches no on-disk file -> hard error."""
    repo = _stub_repo(tmp_path)
    _add_deferral(
        repo, path_glob="roger/old.json", reason="legacy",
        category=deferral.DeferralCategory.SUPERSEDED,
        replaced_by="roger/never_existed.json",
    )
    _patch_git_ls_files(monkeypatch, set())
    report = audit_deferrals.audit(repo_root=repo)
    assert len(report.errors()) == 1
    assert report.errors()[0].invariant == "superseded_replaced_by_missing"


def test_superseded_replaced_by_present_no_issue(
    tmp_path: Path, monkeypatch,
) -> None:
    repo = _stub_repo(tmp_path)
    successor_dir = repo / "roger"
    successor_dir.mkdir()
    successor = successor_dir / "new.json"
    successor.write_text("{}")
    _add_deferral(
        repo, path_glob="roger/old.json", reason="legacy",
        category=deferral.DeferralCategory.SUPERSEDED,
        replaced_by="roger/new.json",
    )
    _patch_git_ls_files(monkeypatch, set())
    report = audit_deferrals.audit(repo_root=repo)
    assert report.errors() == []
    assert report.warnings() == []


# ---------------------------------------------------------------------------
# frozen_snapshot: compares_to (optional)
# ---------------------------------------------------------------------------

def test_frozen_snapshot_no_compares_to_no_issue(
    tmp_path: Path, monkeypatch,
) -> None:
    """The compares_to field is optional (the __rubric_v1 naming
    convention is self-documenting)."""
    repo = _stub_repo(tmp_path)
    _add_deferral(
        repo, path_glob="roger/snap_v1.json", reason="snapshot",
        category=deferral.DeferralCategory.FROZEN_SNAPSHOT,
    )
    _patch_git_ls_files(monkeypatch, set())
    report = audit_deferrals.audit(repo_root=repo)
    assert report.errors() == []
    assert report.warnings() == []


def test_frozen_snapshot_compares_to_missing_warns(
    tmp_path: Path, monkeypatch,
) -> None:
    repo = _stub_repo(tmp_path)
    _add_deferral(
        repo, path_glob="roger/snap_v1.json", reason="snapshot",
        category=deferral.DeferralCategory.FROZEN_SNAPSHOT,
        compares_to="roger/twin_does_not_exist.json",
    )
    _patch_git_ls_files(monkeypatch, set())
    report = audit_deferrals.audit(repo_root=repo)
    assert len(report.warnings()) == 1
    assert report.warnings()[0].invariant == "frozen_snapshot_twin_missing"


# ---------------------------------------------------------------------------
# uncategorized: explicit error
# ---------------------------------------------------------------------------

def test_uncategorized_entry_errors(tmp_path: Path, monkeypatch) -> None:
    """A v1-schema entry (no category) loads as UNCATEGORIZED.  The
    audit raises so backfill is forced before the entry is
    forgotten."""
    repo = _stub_repo(tmp_path)
    # Bypass the CLI (which warns instead of accepting); emit a v1
    # YAML file directly.
    import yaml as _yaml
    (repo / deferral.DEFERRAL_FILENAME).write_text(_yaml.safe_dump({
        "schema_version": 1,
        "deferrals": [
            {"path_glob": "roger/x.json", "dep_key": None,
             "reason": "needs backfill", "deferred_at": "2026-05-08"},
        ],
    }))
    _patch_git_ls_files(monkeypatch, set())
    report = audit_deferrals.audit(repo_root=repo)
    assert len(report.errors()) == 1
    assert report.errors()[0].invariant == "uncategorized"


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def test_format_markdown_smoke(tmp_path: Path, monkeypatch) -> None:
    repo = _stub_repo(tmp_path)
    _add_deferral(
        repo, path_glob="roger/old.json", reason="legacy",
        category=deferral.DeferralCategory.SUPERSEDED,
        replaced_by="roger/never.json",
    )
    _patch_git_ls_files(monkeypatch, set())
    report = audit_deferrals.audit(repo_root=repo)
    md = audit_deferrals._format_markdown(report)
    assert "# Deferral registry audit" in md
    assert "superseded_replaced_by_missing" in md


def test_strict_mode_exits_nonzero(
    tmp_path: Path, monkeypatch, capsys,
) -> None:
    """With --strict, the CLI exits 1 on any error."""
    repo = _stub_repo(tmp_path)
    _add_deferral(
        repo, path_glob="roger/old.json", reason="legacy",
        category=deferral.DeferralCategory.SUPERSEDED,
        replaced_by="roger/missing.json",
    )
    _patch_git_ls_files(monkeypatch, set())
    monkeypatch.setattr(audit_deferrals, "_REPO_ROOT", repo)
    rc = audit_deferrals.main(["--format", "text", "--strict"])
    assert rc == 1


def test_non_strict_exits_zero_even_on_errors(
    tmp_path: Path, monkeypatch, capsys,
) -> None:
    repo = _stub_repo(tmp_path)
    _add_deferral(
        repo, path_glob="roger/old.json", reason="legacy",
        category=deferral.DeferralCategory.SUPERSEDED,
        replaced_by="roger/missing.json",
    )
    _patch_git_ls_files(monkeypatch, set())
    monkeypatch.setattr(audit_deferrals, "_REPO_ROOT", repo)
    rc = audit_deferrals.main(["--format", "text"])
    assert rc == 0
