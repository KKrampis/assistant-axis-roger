"""Tests for data_analysis/check_arrangements.py and backfill_arrangements.py."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from data_analysis import backfill_arrangements as bf  # noqa: E402
from data_analysis import check_arrangements as chk  # noqa: E402


def _write(d: Path, stem: str, doc: dict) -> None:
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{stem}.json").write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def trait(pos, neg, **extra):
    doc = {"positive_label": pos, "negative_label": neg, "description": f"This means being {pos}.",
           "instruction": [{"pos": "x", "neg": "y"}], "questions": ["q"], "eval_prompt": "e"}
    doc.update(extra)
    return doc


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    t = tmp_path / "traits" / "instructions"
    _write(t, "callous", trait("callous", "compassionate"))
    _write(t, "compassionate", trait("compassionate", "non-compassionate"))
    _write(t, "malicious", trait("malicious", "benign"))
    _write(t, "benign", trait("benign", "malicious"))
    _write(t, "blunt", trait("blunt", "tactful"))
    _write(t, "tactful", trait("tactful", "blunt"))
    _write(t, "calm", trait("calm", "agitated"))            # real word, no file -> unclassified
    _write(t, "curious", trait("curious", "non-curious"))    # placeholder, nothing points -> singleton
    _write(t, "selfish", trait("selfish", "unselfish"))      # in the moral-circle sequence
    _write(t, "cosmopolitan", trait("cosmopolitan", "non-cosmopolitan"))
    _write(t, "conformist", trait("conformist", "non-conformist"))
    _write(t, "contrarian", trait("contrarian", "conformist"))
    _write(t, "nonconformist", trait("nonconformist", "conformist"))
    r = tmp_path / "roles" / "instructions"
    _write(r, "angel", {"description": "An angel.", "instruction": [{"pos": "a"}]})
    _write(r, "demon", {"description": "A demon.", "instruction": [{"pos": "d"}]})
    _write(r, "chef", {"description": "A chef.", "instruction": [{"pos": "c"}]})
    _write(r, "default", {"instruction": [{"pos": ""}]})
    (tmp_path / "pair_list_clean.json").write_text(json.dumps([
        {"pos": "angel", "neg": "demon", "pair_type": "roles"},
        {"pos": "blunt", "neg": "tactful", "pair_type": "traits"},
    ]), encoding="utf-8")
    return tmp_path


class TestBackfill:
    def test_plan_traits(self, corpus: Path):
        plan = bf.plan_traits(corpus)
        kinds = {s: sorted(a.kind for a in arrs) for s, arrs in plan.items()}
        assert kinds["blunt"] == ["pair"] and kinds["tactful"] == ["pair"]
        assert kinds["malicious"] == ["pair", "triangle"]          # pair with benign + triangle
        assert kinds["compassionate"] == ["triangle"] and kinds["callous"] == ["triangle"]
        assert kinds["conformist"] == ["triangle"]
        assert kinds["curious"] == ["singleton"]
        assert kinds["calm"] == []                                  # unclassified
        assert kinds["selfish"] == ["sequence"] and kinds["cosmopolitan"] == ["sequence"]
        seq = next(a for a in plan["selfish"] if a.kind == "sequence")
        assert seq.members == ("selfish", "cosmopolitan")           # corpus order preserved, absent members skipped
        assert seq.note and "guesstimate" in seq.note

    def test_plan_roles(self, corpus: Path):
        plan = bf.plan_roles(corpus, corpus / "pair_list_clean.json")
        assert [a.kind for a in plan["angel"]] == ["pair"]
        assert plan["angel"][0].members == ("angel", "demon")
        assert [a.kind for a in plan["chef"]] == ["singleton"]
        assert [a.kind for a in plan["default"]] == ["singleton"]

    def test_dry_run_writes_nothing(self, corpus: Path, capsys):
        before = {p: p.read_text() for p in corpus.rglob("*.json")}
        assert bf.main(["--data_dir", str(corpus), "--pair_list", str(corpus / "pair_list_clean.json")]) == 0
        after = {p: p.read_text() for p in corpus.rglob("*.json")}
        assert before == after
        out = capsys.readouterr().out
        assert "[DRY RUN] traits" in out and "left unclassified: 1" in out
        assert "in more than one arrangement: malicious" in out
        assert "NOT written" in out

    def test_apply_writes_and_validates(self, corpus: Path):
        args = ["--data_dir", str(corpus), "--pair_list", str(corpus / "pair_list_clean.json"), "--apply", "--quiet"]
        assert bf.main(args) == 0
        blunt = json.loads((corpus / "traits/instructions/blunt.json").read_text())
        assert blunt["arrangement"] == {"kind": "pair", "members": ["blunt", "tactful"]}
        assert list(blunt) == ["positive_label", "negative_label", "description", "arrangement",
                               "instruction", "questions", "eval_prompt"]
        mal = json.loads((corpus / "traits/instructions/malicious.json").read_text())
        assert mal["arrangement"] == [
            {"kind": "pair", "members": ["benign", "malicious"]},
            {"kind": "triangle", "members": ["callous", "compassionate", "malicious"]},
        ]
        assert "arrangement" not in json.loads((corpus / "traits/instructions/calm.json").read_text())
        default = json.loads((corpus / "roles/instructions/default.json").read_text())
        assert list(default) == ["arrangement", "instruction"]
        text = (corpus / "traits/instructions/blunt.json").read_text()
        assert text.endswith("}\n") and '\n  "arrangement": {' in text
        # the checker agrees
        assert chk.main(["--data_dir", str(corpus), "--quiet"]) == 0
        # idempotent: second apply skips everything
        assert bf.main(args) == 0
        assert json.loads((corpus / "traits/instructions/blunt.json").read_text()) == blunt

    def test_overwrite_replaces(self, corpus: Path):
        base = ["--data_dir", str(corpus), "--pair_list", str(corpus / "pair_list_clean.json"), "--quiet"]
        assert bf.main(base + ["--apply"]) == 0
        p = corpus / "traits/instructions/curious.json"
        doc = json.loads(p.read_text()); doc["arrangement"] = {"kind": "singleton", "note": "hand-edited"}
        _write(p.parent, "curious", doc)
        assert bf.main(base + ["--apply"]) == 0        # skipped: field present
        assert json.loads(p.read_text())["arrangement"]["note"] == "hand-edited"
        assert bf.main(base + ["--apply", "--overwrite"]) == 0
        assert json.loads(p.read_text())["arrangement"] == {"kind": "singleton"}


class TestCheck:
    def test_exit_codes(self, corpus: Path, capsys):
        assert chk.main(["--data_dir", str(corpus)]) == 0          # nothing classified yet: clean
        out = capsys.readouterr().out
        assert "unclassified: 13" in out
        assert chk.main(["--data_dir", str(corpus), "--kinds", "traits", "--list-unclassified"]) == 0
        assert "unclassified: benign blunt callous" in capsys.readouterr().out
        # break something: declare a pair the labels contradict
        t = corpus / "traits" / "instructions"
        _write(t, "calm", trait("calm", "agitated", arrangement={"kind": "pair", "members": ["calm", "curious"]}))
        _write(t, "curious", trait("curious", "non-curious", arrangement={"kind": "pair", "members": ["calm", "curious"]}))
        assert chk.main(["--data_dir", str(corpus), "--quiet"]) == 1
        assert "not reciprocal by negative_label" in capsys.readouterr().out
        assert chk.main(["--data_dir", str(corpus / "nowhere")]) == 2
