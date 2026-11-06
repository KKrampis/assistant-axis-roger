"""Tests for data_analysis/seed_entities.py (queue-driven seeding helper)."""

import json
from pathlib import Path

import pytest

import data_analysis.seed_entities as se


def _queue(tmp_path: Path, entries):
    q = {"_meta": {}, "entries": entries}
    p = tmp_path / "seed_queue.json"
    p.write_text(json.dumps(q))
    return p


def _data_dir(tmp_path: Path):
    d = tmp_path / "data"
    for et in ("traits", "roles"):
        (d / et / "instructions").mkdir(parents=True)
    return d


def _entry(**kw):
    base = {"stem": "rationalizing", "label": "rationalizing", "entity_type": "trait",
            "chunk": "3", "sub_chunk": "agentic", "pairing": "pair", "partner": "intellectually_honest",
            "description": "This means reasoning backward from what one already wants.",
            "status": "ready", "tags": [], "source": None}
    base.update(kw)
    return base


class TestClassifyCheck:
    REG = {"honest", "objective", "open_minded", "rationalizing"}

    def test_one_word_intended(self):
        v = se.classify_check("intellectually honest", "intellectually_honest", self.REG | {"intellectually_honest"})
        assert v["category"] == "nice" and v["intended_hit"]

    def test_one_word_known_not_intended(self):
        assert se.classify_check("honest", "intellectually_honest", self.REG)["category"] == "mismatch"

    def test_one_word_unknown(self):
        assert se.classify_check("clear-eyed", "intellectually_honest", self.REG)["category"] == "open"

    def test_several_one_known(self):
        v = se.classify_check("honest|clear-eyed", "intellectually_honest", self.REG)
        assert v["category"] == "nearly_nice" and v["known"] == ["honest"]

    def test_intended_among_several_is_confirmed(self):
        v = se.classify_check("calm|composed", "calm", self.REG | {"calm", "composed"})
        assert v["category"] == "nice_with_alternatives" and v["intended_hit"] and v["known"] == ["calm", "composed"]

    def test_several_many_known(self):
        assert se.classify_check("honest|objective", None, self.REG)["category"] == "nasty"

    def test_several_none_known(self):
        assert se.classify_check("clear-eyed|self-honest", None, self.REG)["category"] == "open"

    def test_error_sentinel(self):
        assert se.classify_check("ERROR", None, self.REG)["category"] == "error"

    def test_candidates_are_normalised(self):
        v = se.classify_check("Open-Minded | risk averse", None, self.REG)
        assert v["candidates"] == ["open_minded", "risk_averse"]


class TestSeedDocument:
    def test_trait_non_x_default(self, tmp_path):
        q = {"entries": [_entry()]}
        doc = se.seed_document(q["entries"][0], q, _data_dir(tmp_path))
        assert doc["positive_label"] == "rationalizing"
        assert doc["negative_label"] == "non-rationalizing"
        assert doc["arrangement"] == {"kind": "singleton"}
        assert list(doc)[:3] == ["positive_label", "negative_label", "description"]

    def test_trait_pair_by_construction_uses_partner_label(self, tmp_path):
        a = _entry(stem="openness_big_five", label="openness (Big Five)", partner="closedness_big_five",
                   pair_by_construction=True, source="Big Five")
        b = _entry(stem="closedness_big_five", label="closedness (Big Five)", partner="openness_big_five")
        q = {"entries": [a, b]}
        doc = se.seed_document(a, q, _data_dir(tmp_path))
        assert doc["negative_label"] == "closedness (Big Five)"
        assert doc["source"] == "Big Five"
        assert doc["arrangement"] == {"kind": "pair", "members": ["closedness_big_five", "openness_big_five"]}

    def test_partner_label_from_corpus_file(self, tmp_path):
        d = _data_dir(tmp_path)
        (d / "traits" / "instructions" / "existing.json").write_text(json.dumps({"positive_label": "existing one"}))
        a = _entry(partner="existing", pair_by_construction=True)
        assert se.seed_document(a, {"entries": [a]}, d)["negative_label"] == "existing one"

    def test_set_arrangement_written_at_seed(self, tmp_path):
        a = _entry(stem="hindu", label="hindu", pairing="set", partner=None,
                   arrangement_members=["muslim", "hindu", "sikh"])
        doc = se.seed_document(a, {"entries": [a]}, _data_dir(tmp_path))
        assert doc["arrangement"] == {"kind": "set", "members": ["hindu", "muslim", "sikh"]}

    def test_ring_keeps_order(self, tmp_path):
        a = _entry(stem="b", label="b", pairing="ring", partner=None, arrangement_members=["c", "a", "b"])
        assert se.seed_document(a, {"entries": [a]}, _data_dir(tmp_path))["arrangement"]["members"] == ["c", "a", "b"]

    def test_role_document(self, tmp_path):
        r = _entry(stem="gamekeeper", label="gamekeeper", entity_type="role", pairing="singleton", partner=None,
                   description="A gamekeeper is someone who ...", tags=["x"])
        doc = se.seed_document(r, {"entries": [r]}, _data_dir(tmp_path))
        assert "positive_label" not in doc and "negative_label" not in doc
        assert doc["tags"] == ["x"] and doc["arrangement"] == {"kind": "singleton"}

    def test_missing_description_raises(self, tmp_path):
        with pytest.raises(ValueError):
            se.seed_document(_entry(description=""), {"entries": []}, _data_dir(tmp_path))


class TestWriteCommand:
    def test_write_creates_file_and_advances_status(self, tmp_path, monkeypatch):
        d = _data_dir(tmp_path)
        qp = _queue(tmp_path, [_entry()])
        monkeypatch.setattr(se, "run_tool", lambda *a, **k: None)
        rc = se.main(["--queue", str(qp), "--data-dir", str(d), "write", "--stems", "rationalizing"])
        assert rc == 0
        f = d / "traits" / "instructions" / "rationalizing.json"
        assert f.exists()
        assert json.loads(f.read_text())["negative_label"] == "non-rationalizing"
        assert json.loads(qp.read_text())["entries"][0]["status"] == "seeded"

    def test_write_refuses_overwrite(self, tmp_path, monkeypatch, capsys):
        d = _data_dir(tmp_path)
        f = d / "traits" / "instructions" / "rationalizing.json"
        f.write_text(json.dumps({"positive_label": "rationalizing", "description": "old"}))
        qp = _queue(tmp_path, [_entry()])
        monkeypatch.setattr(se, "run_tool", lambda *a, **k: None)
        se.main(["--queue", str(qp), "--data-dir", str(d), "write", "--stems", "rationalizing"])
        assert json.loads(f.read_text())["description"] == "old"
        assert "exists" in capsys.readouterr().err
        assert json.loads(qp.read_text())["entries"][0]["status"] == "ready"

    def test_chunk_mode_skips_candidates(self, tmp_path, monkeypatch):
        d = _data_dir(tmp_path)
        qp = _queue(tmp_path, [_entry(status="candidate"), _entry(stem="other", label="other")])
        monkeypatch.setattr(se, "run_tool", lambda *a, **k: None)
        se.main(["--queue", str(qp), "--data-dir", str(d), "write", "--chunk", "3"])
        assert not (d / "traits" / "instructions" / "rationalizing.json").exists()
        assert (d / "traits" / "instructions" / "other.json").exists()

    def test_dry_run_writes_nothing(self, tmp_path):
        d = _data_dir(tmp_path)
        qp = _queue(tmp_path, [_entry()])
        se.main(["--queue", str(qp), "--data-dir", str(d), "write", "--stems", "rationalizing", "--dry-run"])
        assert not (d / "traits" / "instructions" / "rationalizing.json").exists()
        assert json.loads(qp.read_text())["entries"][0]["status"] == "ready"


class TestGenerateCommand:
    def test_commands_split_by_type(self):
        cmds = se.generation_commands([_entry(), _entry(stem="gamekeeper", entity_type="role")])
        assert cmds[0][:2] == ["data_analysis/regenerate_trait_instructions.py", "--traits"] and "rationalizing" in cmds[0]
        assert cmds[1][:2] == ["data_analysis/regenerate_role_instructions.py", "--roles"] and "gamekeeper" in cmds[1]

    def test_dry_run_prints_commands_without_running(self, tmp_path, capsys):
        d = _data_dir(tmp_path)
        qp = _queue(tmp_path, [_entry(status="seeded")])
        rc = se.main(["--queue", str(qp), "--data-dir", str(d), "generate", "--chunk", "3", "--dry-run"])
        assert rc == 0
        err = capsys.readouterr().err
        assert "regenerate_trait_instructions.py --traits rationalizing" in err and "estimated $0.03" in err

    def test_expensive_batch_refused_without_flag(self, tmp_path):
        d = _data_dir(tmp_path)
        entries = [_entry(stem=f"t{i}", label=f"t{i}", status="seeded") for i in range(700)]
        qp = _queue(tmp_path, entries)
        with pytest.raises(SystemExit, match="exceeds"):
            se.main(["--queue", str(qp), "--data-dir", str(d), "generate", "--chunk", "3"])

    def test_marks_generated_only_when_file_has_instructions(self, tmp_path, monkeypatch):
        d = _data_dir(tmp_path)
        qp = _queue(tmp_path, [_entry(status="seeded"), _entry(stem="other", label="other", status="seeded")])
        (d / "traits" / "instructions" / "rationalizing.json").write_text(json.dumps({"instruction": [{"pos": "x", "neg": "y"}]}))
        (d / "traits" / "instructions" / "other.json").write_text(json.dumps({"description": "seed only"}))

        class Res:
            returncode = 0
        monkeypatch.setattr(se, "run_tool", lambda *a, **k: Res())
        se.main(["--queue", str(qp), "--data-dir", str(d), "generate", "--chunk", "3"])
        st = {e["stem"]: e["status"] for e in json.loads(qp.read_text())["entries"]}
        assert st == {"rationalizing": "generated", "other": "seeded"}


class TestPairCommand:
    def test_pair_sets_labels_and_arrangement(self, tmp_path, monkeypatch):
        d = _data_dir(tmp_path)
        td = d / "traits" / "instructions"
        td.joinpath("rationalizing.json").write_text(json.dumps({"positive_label": "rationalizing", "negative_label": "non-rationalizing", "description": "d", "arrangement": {"kind": "singleton"}}))
        td.joinpath("intellectually_honest.json").write_text(json.dumps({"positive_label": "intellectually honest", "negative_label": "non-intellectually honest", "description": "d"}))
        qp = _queue(tmp_path, [_entry(status="checked"), _entry(stem="intellectually_honest", label="intellectually honest", partner="rationalizing", status="checked")])
        calls = []

        class Res:
            returncode = 0
        monkeypatch.setattr(se, "run_tool", lambda cmd, **k: (calls.append(cmd), Res())[1])
        rc = se.main(["--queue", str(qp), "--data-dir", str(d), "pair", "--a", "rationalizing", "--b", "intellectually_honest", "--regenerate", "both"])
        assert rc == 0
        a = json.loads(td.joinpath("rationalizing.json").read_text())
        b = json.loads(td.joinpath("intellectually_honest.json").read_text())
        assert a["negative_label"] == "intellectually honest" and b["negative_label"] == "rationalizing"
        assert a["arrangement"] == b["arrangement"] == {"kind": "pair", "members": ["intellectually_honest", "rationalizing"]}
        assert any("--instructions-only" in c for c in calls)
        assert all(e["status"] == "paired" for e in json.loads(qp.read_text())["entries"])

    def test_pair_dry_run_changes_nothing(self, tmp_path, monkeypatch):
        d = _data_dir(tmp_path)
        td = d / "traits" / "instructions"
        td.joinpath("a.json").write_text(json.dumps({"positive_label": "a", "negative_label": "non-a"}))
        td.joinpath("b.json").write_text(json.dumps({"positive_label": "b", "negative_label": "non-b"}))
        qp = _queue(tmp_path, [])
        monkeypatch.setattr(se, "run_tool", lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not run")))
        se.main(["--queue", str(qp), "--data-dir", str(d), "pair", "--a", "a", "--b", "b", "--dry-run"])
        assert json.loads(td.joinpath("a.json").read_text())["negative_label"] == "non-a"


class TestStatus:
    def test_summary_counts(self, tmp_path, capsys):
        d = _data_dir(tmp_path)
        qp = _queue(tmp_path, [_entry(), _entry(stem="x", label="x", status="candidate", chunk="1"), _entry(stem="y", label="y", status="tbd", chunk="7")])
        se.main(["--queue", str(qp), "--data-dir", str(d), "status"])
        out = capsys.readouterr().out
        assert "3 entries" in out and '"candidate": 1' in out and "chunk    7" in out


class TestMergePairArrangement:
    PAIR = {"kind": "pair", "members": ["altruistic", "selfish"]}

    def test_replaces_missing_and_singleton(self):
        assert se.merge_pair_arrangement(None, self.PAIR) == self.PAIR
        assert se.merge_pair_arrangement({"kind": "singleton"}, self.PAIR) == self.PAIR

    def test_replaces_singleton_with_note(self):
        assert se.merge_pair_arrangement({"kind": "singleton", "note": "why"}, self.PAIR) == self.PAIR
        assert se.merge_pair_arrangement([{"kind": "singleton", "note": "why"}], self.PAIR) == self.PAIR

    def test_keeps_identical_pair(self):
        assert se.merge_pair_arrangement({"kind": "pair", "members": ["altruistic", "selfish"]}, self.PAIR) == self.PAIR

    def test_appends_to_sequence_member(self):
        seq = {"kind": "sequence", "members": ["selfish", "clannish", "universalist"]}
        assert se.merge_pair_arrangement(seq, self.PAIR) == [seq, self.PAIR]

    def test_appends_to_existing_list_once(self):
        seq = {"kind": "sequence", "members": ["selfish", "clannish"]}
        out = se.merge_pair_arrangement([seq, self.PAIR], self.PAIR)
        assert out == [seq, self.PAIR]


class TestRenameCommand:
    def _setup(self, tmp_path):
        d = _data_dir(tmp_path); td = d / "traits" / "instructions"
        td.joinpath("slothful.json").write_text(json.dumps({"positive_label": "slothful", "negative_label": "industrious", "description": "d", "instruction": [{"pos": "x", "neg": "y"}]}))
        qp = _queue(tmp_path, [_entry(stem="industrious", label="industrious", partner="slothful", status="checked")])
        return d, td, qp

    def test_rename_moves_file_relabels_and_updates_partner(self, tmp_path, monkeypatch):
        d, td, qp = self._setup(tmp_path)

        class Res:
            returncode = 0
            stdout = json.dumps({"lazy": {"negative_label": "industrious", "antonym_score": 4, "reasoning": "r"}, "industrious": {"negative_label": "lazy", "antonym_score": 4, "reasoning": "r"}})
            stderr = ""
        calls = []
        monkeypatch.setattr(se, "run_tool", lambda cmd, **k: (calls.append(cmd), Res())[1])
        rc = se.main(["--queue", str(qp), "--data-dir", str(d), "rename", "--old", "slothful", "--new", "lazy", "--partner", "industrious"])
        assert rc == 0
        assert not td.joinpath("slothful.json").exists() and td.joinpath("lazy.json").exists()
        nd = json.loads(td.joinpath("lazy.json").read_text())
        assert nd["positive_label"] == "lazy" and nd["negative_label"] == "industrious" and nd["renamed_from"]["stem"] == "slothful"
        assert any("regenerate_trait_instructions.py" in c[0] and "lazy" in c for c in calls)
        q = json.loads(qp.read_text()); e = q["entries"][0]
        assert e["partner"] == "lazy" and e["check_result"]["category"] == "nice"

    def test_rename_refuses_taken_name(self, tmp_path):
        d, td, qp = self._setup(tmp_path)
        td.joinpath("lazy.json").write_text("{}")
        with pytest.raises(SystemExit, match="not free"):
            se.main(["--queue", str(qp), "--data-dir", str(d), "rename", "--old", "slothful", "--new", "lazy"])

    def test_rename_refuses_queued_name_without_force(self, tmp_path):
        d, td, qp = self._setup(tmp_path)
        q = json.loads(qp.read_text()); q["entries"].append(_entry(stem="lazy", label="lazy", status="candidate")); qp.write_text(json.dumps(q))
        with pytest.raises(SystemExit, match="queued"):
            se.main(["--queue", str(qp), "--data-dir", str(d), "rename", "--old", "slothful", "--new", "lazy"])

    def test_rename_dry_run_changes_nothing(self, tmp_path):
        d, td, qp = self._setup(tmp_path)
        se.main(["--queue", str(qp), "--data-dir", str(d), "rename", "--old", "slothful", "--new", "lazy", "--dry-run"])
        assert td.joinpath("slothful.json").exists() and not td.joinpath("lazy.json").exists()


class TestDuplicateLiveStems:
    def test_finds_two_live_copies_and_ignores_parked(self):
        q = {"entries": [_entry(stem="x", label="x", status="candidate", partner="a"), _entry(stem="x", label="x", status="ready", partner="b"),
                         _entry(stem="y", label="y", status="candidate"), _entry(stem="y", label="y", status="superseded")]}
        d = se.duplicate_live_stems(q)
        assert list(d) == ["x (trait)"] and len(d["x (trait)"]) == 2

    def test_status_warns(self, tmp_path, capsys):
        d = _data_dir(tmp_path)
        qp = _queue(tmp_path, [_entry(stem="x", label="x"), _entry(stem="x", label="x", partner="other")])
        se.main(["--queue", str(qp), "--data-dir", str(d), "status"])
        assert "WARNING duplicate live stem x (trait)" in capsys.readouterr().err


class TestCheckHistory:
    """Every antonym check is kept: candidates plus the instructions it read."""

    DOC = {"positive_label": "rooted", "negative_label": "non-rooted",
           "description": "This means being from somewhere.",
           "instruction": [{"pos": "You are rooted.", "neg": "You are rootless."}],
           "generator": {"model": "claude-sonnet-4-6"}}

    def _record(self, returned, phase="check"):
        answer = {"negative_label": returned, "antonym_score": 4, "reasoning": "r"}
        verdict = se.classify_check(returned, "deracinated", {"deracinated", "cosmopolitan"})
        return se.check_history_record("rooted", self.DOC, answer, verdict, "deracinated", phase)

    def test_record_keeps_candidates_and_instructions(self):
        rec = self._record("rootless|cosmopolitan")
        assert rec["candidates"] == ["rootless", "cosmopolitan"]
        assert rec["known"] == ["cosmopolitan"] and rec["intended_hit"] is False
        assert rec["negative_label_at_check"] == "non-rooted"
        assert rec["instructions"] == self.DOC["instruction"]
        assert rec["description"] == self.DOC["description"]

    def test_append_never_overwrites(self, tmp_path):
        data = _data_dir(tmp_path)
        assert se.append_check_history(data, [self._record("rootless|cosmopolitan")]) == 1
        assert se.append_check_history(data, [self._record("deracinated|rootless", phase="resample")]) == 1
        lines = se.check_history_path(data).read_text().splitlines()
        assert [json.loads(x)["returned"] for x in lines] == ["rootless|cosmopolitan", "deracinated|rootless"]
        assert json.loads(lines[1])["intended_hit"] is True and json.loads(lines[1])["phase"] == "resample"

    def test_history_sits_beside_the_corpus_not_in_it(self, tmp_path):
        data = _data_dir(tmp_path)
        se.append_check_history(data, [self._record("rootless")])
        assert se.check_history_path(data).parent == data / "traits"
        assert list((data / "traits" / "instructions").glob("*")) == []

    def test_empty_append_writes_nothing(self, tmp_path):
        data = _data_dir(tmp_path)
        assert se.append_check_history(data, []) == 0
        assert not se.check_history_path(data).exists()

    def test_queue_entry_keeps_every_answer(self):
        e = _entry()
        se.note_check_answer(e, "rootless|cosmopolitan", "check")
        se.note_check_answer(e, "deracinated", "resample")
        assert [a["returned"] for a in e["check_answers"]] == ["rootless|cosmopolitan", "deracinated"]
