"""Tests for assistant_axis.arrangements (the ``arrangement`` field)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from assistant_axis import arrangements as arr
from assistant_axis.arrangements import (
    Arrangement, ArrangementError, arrangement_to_json, canonical_kind,
    expected_size, field_to_json, parse_arrangement, parse_field,
    reciprocal_pairs, summarize_corpus, validate_corpus,
)


# ---------------------------------------------------------------------------
# kinds and sizes
# ---------------------------------------------------------------------------

class TestKinds:
    @pytest.mark.parametrize("given,canon", [
        ("pair", "pair"), ("PAIR", "pair"), (" triangle ", "triangle"),
        ("2-simplex", "triangle"), ("3-simplex", "tetrahedron"), ("4-simplex", "4-simplex"),
        ("1-orthoplex", "pair"), ("1-cube", "pair"), ("1-simplex", "pair"), ("0-simplex", "singleton"),
        ("2-cube", "square"), ("2-orthoplex", "square"), ("3-cube", "cube"),
        ("3-orthoplex", "octahedron"), ("6-orthoplex", "6-orthoplex"), ("4-cube", "4-cube"),
        ("ring", "ring"), ("tree", "tree"), ("map", "map"), ("sequence", "sequence"), ("set", "set"),
    ])
    def test_canonical(self, given, canon):
        assert canonical_kind(given) == canon

    @pytest.mark.parametrize("bad", ["octagon", "6-octahedron", "0-cube", "2-orthoplex-thing", "", 3])
    def test_unknown_kind(self, bad):
        with pytest.raises(ArrangementError):
            canonical_kind(bad)

    @pytest.mark.parametrize("kind,size", [
        ("singleton", (0, 0)), ("pair", (2, 2)), ("triangle", (3, 3)), ("tetrahedron", (4, 4)),
        ("5-simplex", (6, 6)), ("square", (4, 4)), ("cube", (8, 8)), ("4-cube", (16, 16)),
        ("octahedron", (6, 6)), ("6-orthoplex", (12, 12)), ("ring", (3, None)),
        ("tree", (2, None)), ("map", (2, None)), ("sequence", (2, None)), ("set", (2, None)),
    ])
    def test_expected_size(self, kind, size):
        assert expected_size(kind) == size

    def test_ordered_and_pairwise(self):
        assert arr.is_ordered("sequence") and arr.is_ordered("ring")
        assert not arr.is_ordered("set")
        assert arr.is_pairwise("octahedron") and arr.is_pairwise("6-orthoplex")
        assert not arr.is_pairwise("square") and not arr.is_pairwise("4-cube")


# ---------------------------------------------------------------------------
# parsing one object
# ---------------------------------------------------------------------------

class TestParse:
    def test_pair(self):
        a = parse_arrangement({"kind": "pair", "members": ["callous", "compassionate"]})
        assert a.kind == "pair" and a.members == ("callous", "compassionate")

    def test_singleton_needs_no_members(self):
        assert parse_arrangement({"kind": "singleton"}).members == ()

    def test_wrong_size(self):
        with pytest.raises(ArrangementError, match="expects 3 members"):
            parse_arrangement({"kind": "triangle", "members": ["a", "b"]})
        with pytest.raises(ArrangementError, match="at least 3"):
            parse_arrangement({"kind": "ring", "members": ["a", "b"]})

    def test_labels_rejected(self):
        with pytest.raises(ArrangementError, match="file-name form"):
            parse_arrangement({"kind": "pair", "members": ["systems-thinker", "analytical"]})

    def test_duplicates_rejected(self):
        with pytest.raises(ArrangementError, match="duplicate"):
            parse_arrangement({"kind": "pair", "members": ["a", "a"]})

    def test_axes_must_name_members(self):
        with pytest.raises(ArrangementError, match="non-member"):
            parse_arrangement({"kind": "square", "members": ["a", "b", "c", "d"], "axes": [["a", "z"]]})
        a = parse_arrangement({"kind": "square", "members": ["a", "b", "c", "d"], "axes": [["a", "b"], ["c", "d"]]})
        assert a.axes == (("a", "b"), ("c", "d"))

    def test_tree_keys_only_on_tree(self):
        with pytest.raises(ArrangementError, match="only valid for kind 'tree'"):
            parse_arrangement({"kind": "set", "members": ["a", "b"], "parent": "a"})
        t = parse_arrangement({"kind": "tree", "members": ["a", "b"], "parent": None, "children": ["b"]})
        assert t.parent is None and t.children == ("b",)

    def test_extras_preserved_and_serialised_last(self):
        a = parse_arrangement({"kind": "set", "members": ["a", "b"], "confirmed": True, "note": "n"})
        assert a.extra == {"confirmed": True}
        js = arrangement_to_json(a)
        assert list(js) == ["kind", "members", "note", "confirmed"]

    def test_field_object_or_list(self):
        assert len(parse_field({"kind": "singleton"})) == 1
        assert len(parse_field([{"kind": "singleton"}, {"kind": "pair", "members": ["a", "b"]}])) == 2
        assert parse_field(None) == []
        with pytest.raises(ArrangementError, match="empty"):
            parse_field([])

    def test_round_trip(self):
        objs = [{"kind": "pair", "members": ["a", "b"]}, {"kind": "triangle", "members": ["a", "b", "c"], "source": "x"}]
        arrs = parse_field(objs)
        assert field_to_json(arrs) == objs
        assert field_to_json(arrs[:1]) == objs[0]


# ---------------------------------------------------------------------------
# corpus validation
# ---------------------------------------------------------------------------

def _write(d: Path, stem: str, doc: dict) -> None:
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{stem}.json").write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")


def trait(pos, neg, arrangement=None):
    doc = {"positive_label": pos, "negative_label": neg, "description": f"This means being {pos}."}
    if arrangement is not None:
        doc["arrangement"] = arrangement
    return doc


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    t = tmp_path / "traits" / "instructions"
    pair = {"kind": "pair", "members": ["a", "b"]}
    _write(t, "a", trait("a", "b", pair))
    _write(t, "b", trait("b", "a", pair))
    tri = {"kind": "triangle", "members": ["c", "d", "e"]}
    _write(t, "c", trait("c", "non-c", tri))
    _write(t, "d", trait("d", "c", tri))
    _write(t, "e", trait("e", "c", tri))
    _write(t, "f", trait("f", "non-f", {"kind": "singleton"}))
    _write(t, "g", trait("g", "nonexistent"))  # unclassified, real-word label
    seq = {"kind": "sequence", "members": ["h", "i", "j"], "note": "order proposed"}
    for s in ("h", "i", "j"):
        _write(t, s, trait(s, f"non-{s}", seq))
    r = tmp_path / "roles" / "instructions"
    rp = {"kind": "pair", "members": ["angel", "demon"]}
    _write(r, "angel", {"description": "An angel.", "arrangement": rp})
    _write(r, "demon", {"description": "A demon.", "arrangement": rp})
    _write(r, "chef", {"description": "A chef.", "arrangement": {"kind": "singleton"}})
    _write(r, "default", {"instruction": [{"pos": ""}]})
    return tmp_path


class TestValidateCorpus:
    def test_clean_corpus(self, corpus: Path):
        assert validate_corpus(corpus, "traits") == []
        assert validate_corpus(corpus, "roles") == []

    def test_reciprocal_pairs(self, corpus: Path):
        recs = arr.load_corpus_arrangements(corpus, "traits")
        assert reciprocal_pairs(recs) == [("a", "b")]

    def test_summary(self, corpus: Path):
        s = summarize_corpus(corpus, "traits")
        assert s["total"] == 10
        assert s["by_kind"] == {"pair": 2, "sequence": 3, "singleton": 1, "triangle": 3}
        assert s["unclassified"] == ["g"]
        assert s["malformed"] == []
        r = summarize_corpus(corpus, "roles")
        assert r["unclassified"] == ["default"] and r["by_kind"] == {"pair": 2, "singleton": 1}

    def test_member_missing_from_corpus(self, corpus: Path):
        t = corpus / "traits" / "instructions"
        _write(t, "k", trait("k", "non-k", {"kind": "set", "members": ["k", "zzz"]}))
        msgs = [str(p) for p in validate_corpus(corpus, "traits")]
        assert any("members not in corpus: zzz" in m for m in msgs)

    def test_member_does_not_reciprocate(self, corpus: Path):
        t = corpus / "traits" / "instructions"
        # e drops the triangle
        _write(t, "e", trait("e", "c"))
        msgs = [str(p) for p in validate_corpus(corpus, "traits")]
        assert any("member e does not record the same arrangement" in m for m in msgs)

    def test_self_must_be_member_and_sorted(self, corpus: Path):
        t = corpus / "traits" / "instructions"
        _write(t, "f", trait("f", "non-f", {"kind": "set", "members": ["h", "i"]}))
        msgs = [str(p) for p in validate_corpus(corpus, "traits")]
        assert any("do not include the entity itself" in m for m in msgs)
        _write(t, "f", trait("f", "non-f", {"kind": "set", "members": ["i", "f"]}))
        msgs = [str(p) for p in validate_corpus(corpus, "traits")]
        assert any("not sorted" in m for m in msgs)

    def test_sequence_order_is_content(self, corpus: Path):
        t = corpus / "traits" / "instructions"
        _write(t, "j", trait("j", "non-j", {"kind": "sequence", "members": ["j", "h", "i"]}))
        msgs = [str(p) for p in validate_corpus(corpus, "traits")]
        assert any("traits/h: sequence: member j does not record the same arrangement" == m for m in msgs)

    def test_pair_must_match_negative_labels(self, corpus: Path):
        t = corpus / "traits" / "instructions"
        # declare a pair the labels do not support
        bad = {"kind": "pair", "members": ["f", "g"]}
        _write(t, "f", trait("f", "non-f", bad))
        _write(t, "g", trait("g", "nonexistent", bad))
        msgs = [str(p) for p in validate_corpus(corpus, "traits")]
        assert any("not reciprocal by negative_label" in m for m in msgs)

    def test_singleton_needs_non_x_label(self, corpus: Path):
        t = corpus / "traits" / "instructions"
        # g's label is a real word with no file: unclassified is fine, singleton is not
        _write(t, "g", trait("g", "nonexistent", {"kind": "singleton"}))
        msgs = [str(p) for p in validate_corpus(corpus, "traits")]
        assert any("traits/g: singleton with a real-word negative_label 'nonexistent'" in m for m in msgs)
        # and a one-way pointer at a trait paired elsewhere is the same case
        _write(t, "g", trait("g", "a", {"kind": "singleton"}))
        msgs = [str(p) for p in validate_corpus(corpus, "traits")]
        assert any("traits/g: singleton with a real-word negative_label 'a'" in m for m in msgs)
        # the non-X singleton f stays clean
        assert not any(m.startswith("traits/f") for m in msgs)

    def test_label_pair_must_be_recorded(self, corpus: Path):
        t = corpus / "traits" / "instructions"
        # a and b reciprocal by label, but a says singleton
        _write(t, "a", trait("a", "b", {"kind": "singleton"}))
        msgs = [str(p) for p in validate_corpus(corpus, "traits")]
        assert any("traits/a: clean pair with b by negative_label is not recorded" in m for m in msgs)
        # unclassified is allowed: drop the field entirely
        _write(t, "a", trait("a", "b"))
        msgs = [str(p) for p in validate_corpus(corpus, "traits")]
        assert not any("traits/a:" in m for m in msgs)

    def test_role_pairs_need_no_labels(self, corpus: Path):
        assert validate_corpus(corpus, "roles") == []

    def test_octahedron_members_must_pair_up(self, tmp_path: Path):
        t = tmp_path / "traits" / "instructions"
        stems = ["p1", "p2", "q1", "q2", "r1", "r2"]
        octa = {"kind": "octahedron", "members": sorted(stems)}
        for s in stems:
            partner = s[0] + ("2" if s.endswith("1") else "1")
            _write(t, s, trait(s, partner, octa))
        assert validate_corpus(tmp_path, "traits") == []
        # break one reciprocal link
        _write(t, "r2", trait("r2", "non-r2", octa))
        msgs = [str(p) for p in validate_corpus(tmp_path, "traits")]
        assert any("not in reciprocal pairs: r1, r2" in m for m in msgs)

    def test_square_allows_both_styles(self, tmp_path: Path):
        t = tmp_path / "traits" / "instructions"
        sq = {"kind": "square", "members": ["w", "x", "y", "z"]}
        for s in "wxyz":  # corner-style: no reciprocal labels at all
            _write(t, s, trait(s, f"non-{s}", sq))
        assert validate_corpus(tmp_path, "traits") == []

    def test_tree_links(self, tmp_path: Path):
        t = tmp_path / "traits" / "instructions"
        members = ["child1", "child2", "root"]
        _write(t, "root", trait("root", "non-root", {"kind": "tree", "members": members, "parent": None, "children": ["child1", "child2"]}))
        _write(t, "child1", trait("child1", "non-child1", {"kind": "tree", "members": members, "parent": "root", "children": []}))
        _write(t, "child2", trait("child2", "non-child2", {"kind": "tree", "members": members, "parent": "root", "children": []}))
        assert validate_corpus(tmp_path, "traits") == []
        _write(t, "child2", trait("child2", "non-child2", {"kind": "tree", "members": members, "parent": None, "children": []}))
        msgs = [str(p) for p in validate_corpus(tmp_path, "traits")]
        assert any("does not name root as parent" in m for m in msgs)
        assert any("expected exactly one root, found 2" in m for m in msgs)

    def test_multiple_arrangements_per_entity(self, corpus: Path):
        t = corpus / "traits" / "instructions"
        # b is in the a/b pair and also in a set with f
        s = {"kind": "set", "members": ["b", "f"]}
        _write(t, "b", trait("b", "a", [{"kind": "pair", "members": ["a", "b"]}, s]))
        _write(t, "f", trait("f", "non-f", s))
        assert validate_corpus(corpus, "traits") == []
        assert summarize_corpus(corpus, "traits")["by_kind"]["set"] == 2

    def test_malformed_reported_not_raised(self, corpus: Path):
        t = corpus / "traits" / "instructions"
        _write(t, "f", trait("f", "non-f", {"kind": "hexagon", "members": ["f"]}))
        msgs = [str(p) for p in validate_corpus(corpus, "traits")]
        assert any("malformed arrangement: unknown arrangement kind" in m for m in msgs)
        assert summarize_corpus(corpus, "traits")["malformed"] == ["f"]


def test_real_corpus_is_consistent():
    """Guard: the checked-in corpus must pass validation (unclassified is fine)."""
    repo_data = Path(__file__).resolve().parents[2] / "data"
    if not (repo_data / "traits" / "instructions").is_dir():
        pytest.skip("repo data directory not present")
    for etype in ("traits", "roles"):
        problems = validate_corpus(repo_data, etype)
        assert problems == [], "\n".join(str(p) for p in problems[:20])
