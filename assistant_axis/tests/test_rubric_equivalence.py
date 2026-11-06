"""Tests for :mod:`assistant_axis.rubric_equivalence`.

The on-disk registry path is monkey-patched to a per-test
``tmp_path``, so tests don't read or mutate the repo-root
``rubric_equivalences.yaml``.
"""
from __future__ import annotations

import yaml
import pytest

from assistant_axis.rubric_equivalence import (
    RubricEquivalenceEdge,
    REGISTRY_FILENAME,
    append_equivalence,
    edge_covers,
    is_equivalent,
    load_registry,
)


# ---------------------------------------------------------------------------
# RubricEquivalenceEdge invariants
# ---------------------------------------------------------------------------

class TestRubricEquivalenceEdge:
    def test_minimal_edge(self):
        e = RubricEquivalenceEdge(from_rubric="v1", to_rubric="v2")
        assert e.from_rubric == "v1"
        assert e.to_rubric == "v2"
        assert e.modes is None
        assert e.axes is None
        assert e.except_axes is None
        assert e.entity_ids is None
        assert e.except_entity_ids is None

    def test_axes_xor_except_axes_enforced(self):
        with pytest.raises(ValueError, match="axes.*except_axes"):
            RubricEquivalenceEdge(
                from_rubric="v1", to_rubric="v2",
                axes=("a",), except_axes=("b",),
            )

    def test_entity_ids_xor_except_entity_ids_enforced(self):
        with pytest.raises(ValueError, match="entity_ids.*except_entity_ids"):
            RubricEquivalenceEdge(
                from_rubric="v1", to_rubric="v2",
                entity_ids=("patient|R",),
                except_entity_ids=("patient|T",),
            )

    def test_axes_and_entity_ids_compose(self):
        # Axis+entity scope can be combined (intersection); doesn't trip
        # the XOR check.
        e = RubricEquivalenceEdge(
            from_rubric="v1", to_rubric="v2",
            axes=("ax1",), entity_ids=("patient|R",),
        )
        assert e.axes == ("ax1",)
        assert e.entity_ids == ("patient|R",)

    def test_unknown_mode_rejected(self):
        with pytest.raises(ValueError, match="unknown mode"):
            RubricEquivalenceEdge(
                from_rubric="v1", to_rubric="v2",
                modes=("nonexistent_mode",),
            )

    def test_hashable(self):
        # Frozen dataclass => hashable; useful for set membership in
        # tests / dedup.
        e1 = RubricEquivalenceEdge(from_rubric="v1", to_rubric="v2")
        e2 = RubricEquivalenceEdge(from_rubric="v1", to_rubric="v2")
        assert {e1, e2} == {e1}


# ---------------------------------------------------------------------------
# edge_covers
# ---------------------------------------------------------------------------

class TestEdgeCovers:
    def test_unscoped_edge_covers_everything(self):
        e = RubricEquivalenceEdge(from_rubric="v1", to_rubric="v2")
        assert edge_covers(e, axis="any", mode="responses",
                           entity_id="patient|R")
        assert edge_covers(e, axis=None, mode=None, entity_id=None)

    def test_modes_allow_list(self):
        e = RubricEquivalenceEdge(
            from_rubric="v1", to_rubric="v2",
            modes=("responses",),
        )
        assert edge_covers(e, axis="ax", mode="responses",
                           entity_id="patient|R")
        assert not edge_covers(e, axis="ax", mode="descriptions",
                               entity_id="patient|R")
        # Wildcard mode = caller doesn't care; edge applies.
        assert edge_covers(e, axis="ax", mode=None, entity_id="patient|R")

    def test_axes_allow_list(self):
        e = RubricEquivalenceEdge(
            from_rubric="v1", to_rubric="v2",
            axes=("ax1", "ax2"),
        )
        assert edge_covers(e, axis="ax1", mode="responses",
                           entity_id=None)
        assert edge_covers(e, axis="ax2", mode="responses",
                           entity_id=None)
        assert not edge_covers(e, axis="ax3", mode="responses",
                               entity_id=None)
        # Wildcard axis = caller doesn't care.
        assert edge_covers(e, axis=None, mode="responses", entity_id=None)

    def test_except_axes_deny_list(self):
        e = RubricEquivalenceEdge(
            from_rubric="v1", to_rubric="v2",
            except_axes=("ax3",),
        )
        assert edge_covers(e, axis="ax1", mode="responses",
                           entity_id=None)
        assert not edge_covers(e, axis="ax3", mode="responses",
                               entity_id=None)

    def test_entity_ids_allow_list(self):
        e = RubricEquivalenceEdge(
            from_rubric="v1", to_rubric="v2",
            entity_ids=("patient|R", "stoic|T"),
        )
        assert edge_covers(e, axis=None, mode=None, entity_id="patient|R")
        assert edge_covers(e, axis=None, mode=None, entity_id="stoic|T")
        assert not edge_covers(e, axis=None, mode=None, entity_id="patient|T")

    def test_except_entity_ids_deny_list(self):
        e = RubricEquivalenceEdge(
            from_rubric="v1", to_rubric="v2",
            except_entity_ids=("patient|R",),
        )
        assert edge_covers(e, axis=None, mode=None, entity_id="patient|T")
        assert not edge_covers(e, axis=None, mode=None, entity_id="patient|R")

    def test_dimensions_intersect(self):
        # axis=ax1 AND mode=responses AND entity_id=patient|R must
        # all be covered for the edge to apply.
        e = RubricEquivalenceEdge(
            from_rubric="v1", to_rubric="v2",
            axes=("ax1",),
            modes=("responses",),
            entity_ids=("patient|R",),
        )
        assert edge_covers(e, axis="ax1", mode="responses",
                           entity_id="patient|R")
        assert not edge_covers(e, axis="ax2", mode="responses",
                               entity_id="patient|R")
        assert not edge_covers(e, axis="ax1", mode="descriptions",
                               entity_id="patient|R")
        assert not edge_covers(e, axis="ax1", mode="responses",
                               entity_id="patient|T")


# ---------------------------------------------------------------------------
# is_equivalent
# ---------------------------------------------------------------------------

class TestIsEquivalent:
    def test_reflexive(self):
        assert is_equivalent("v3", "v3", registry=[])
        # axis/mode/entity_id irrelevant under reflexivity.
        assert is_equivalent("v3", "v3",
                             axis="anything", mode="responses",
                             entity_id="patient|R",
                             registry=[])

    def test_returns_empty_path_for_reflexive(self):
        found, path = is_equivalent(
            "v3", "v3", registry=[], return_path=True
        )
        assert found is True
        assert path == []

    def test_no_edges_no_equivalence(self):
        assert not is_equivalent("v2", "v3", registry=[])

    def test_direct_edge(self):
        edges = [
            RubricEquivalenceEdge(from_rubric="v2", to_rubric="v3"),
        ]
        assert is_equivalent("v2", "v3", registry=edges)
        # Asymmetric: reverse direction not declared.
        assert not is_equivalent("v3", "v2", registry=edges)

    def test_transitive_two_hops(self):
        edges = [
            RubricEquivalenceEdge(from_rubric="v1", to_rubric="v2"),
            RubricEquivalenceEdge(from_rubric="v2", to_rubric="v3"),
        ]
        assert is_equivalent("v1", "v3", registry=edges)
        found, path = is_equivalent(
            "v1", "v3", registry=edges, return_path=True
        )
        assert found is True
        assert len(path) == 2

    def test_transitive_requires_all_hops_to_match_scope(self):
        # v1 -> v2 only for axis=ax1; v2 -> v3 only for axis=ax2.
        # No transitive path scoped to either axis alone.
        edges = [
            RubricEquivalenceEdge(from_rubric="v1", to_rubric="v2",
                                  axes=("ax1",)),
            RubricEquivalenceEdge(from_rubric="v2", to_rubric="v3",
                                  axes=("ax2",)),
        ]
        assert not is_equivalent("v1", "v3",
                                 axis="ax1", registry=edges)
        assert not is_equivalent("v1", "v3",
                                 axis="ax2", registry=edges)
        # But wildcard axis (caller doesn't care) lets both edges
        # apply -- which is the right semantic, since "wildcard" means
        # "the caller is asking a coarser question than the registry's
        # scoping allows it to refuse."  Caller is responsible for
        # passing a specific axis when they want strict per-axis
        # answers.
        assert is_equivalent("v1", "v3", axis=None, registry=edges)

    def test_axis_scope_applied(self):
        edges = [
            RubricEquivalenceEdge(
                from_rubric="v2", to_rubric="v3",
                except_axes=("ax_multi",),
            ),
        ]
        assert is_equivalent("v2", "v3", axis="ax_single", registry=edges)
        assert not is_equivalent("v2", "v3", axis="ax_multi", registry=edges)

    def test_entity_scope_applied(self):
        edges = [
            RubricEquivalenceEdge(
                from_rubric="v2", to_rubric="v3",
                except_entity_ids=("patient|R", "stoic|R"),
            ),
        ]
        assert is_equivalent("v2", "v3",
                             entity_id="ascetic|R", registry=edges)
        assert not is_equivalent("v2", "v3",
                                 entity_id="patient|R", registry=edges)


# ---------------------------------------------------------------------------
# load_registry / append_equivalence (file I/O)
# ---------------------------------------------------------------------------

class TestRegistryFileIO:
    def test_load_returns_empty_when_file_missing(self, tmp_path):
        assert load_registry(repo_root=tmp_path) == []

    def test_unknown_schema_version_rejected(self, tmp_path):
        path = tmp_path / REGISTRY_FILENAME
        path.write_text(yaml.safe_dump({
            "schema_version": 99,
            "equivalences": [],
        }))
        with pytest.raises(ValueError, match="schema_version"):
            load_registry(repo_root=tmp_path)

    def test_append_creates_file(self, tmp_path):
        edge = append_equivalence(
            from_rubric="v2", to_rubric="v3",
            modes=["responses"],
            except_axes=["systems_thinker_vs_analytical"],
            reason="test reason",
            marked_at="2026-05-11T00:00:00+00:00",
            repo_root=tmp_path,
        )
        assert edge.from_rubric == "v2"
        assert edge.to_rubric == "v3"
        assert (tmp_path / REGISTRY_FILENAME).exists()

    def test_append_then_load_roundtrip(self, tmp_path):
        append_equivalence(
            from_rubric="v2", to_rubric="v3",
            modes=["responses"],
            except_axes=["ax3"],
            reason="r1",
            marked_at="2026-05-11T00:00:00+00:00",
            repo_root=tmp_path,
        )
        append_equivalence(
            from_rubric="v3", to_rubric="v4",
            entity_ids=["patient|R"],
            reason="r2",
            marked_at="2026-05-11T00:00:01+00:00",
            repo_root=tmp_path,
        )
        registry = load_registry(repo_root=tmp_path)
        assert len(registry) == 2
        e1 = registry[0]
        e2 = registry[1]
        assert e1.from_rubric == "v2" and e1.to_rubric == "v3"
        assert e1.modes == ("responses",)
        assert e1.except_axes == ("ax3",)
        assert e1.axes is None
        assert e1.entity_ids is None
        assert e2.from_rubric == "v3" and e2.to_rubric == "v4"
        assert e2.entity_ids == ("patient|R",)
        assert e2.except_entity_ids is None
        assert e2.modes is None

    def test_append_dedup_same_scope(self, tmp_path):
        append_equivalence(
            from_rubric="v2", to_rubric="v3",
            modes=["responses"],
            reason="r1",
            repo_root=tmp_path,
        )
        with pytest.raises(ValueError, match="Duplicate"):
            append_equivalence(
                from_rubric="v2", to_rubric="v3",
                modes=["responses"],
                reason="r2-different-reason",
                repo_root=tmp_path,
            )

    def test_append_distinct_scope_not_dedup(self, tmp_path):
        # Same (from, to) but different scope = distinct edges.
        append_equivalence(
            from_rubric="v2", to_rubric="v3",
            modes=["responses"],
            reason="r1",
            repo_root=tmp_path,
        )
        append_equivalence(
            from_rubric="v2", to_rubric="v3",
            modes=["descriptions"],
            reason="r2",
            repo_root=tmp_path,
        )
        registry = load_registry(repo_root=tmp_path)
        assert len(registry) == 2

    def test_append_rejects_conflicting_scope_kwargs(self, tmp_path):
        with pytest.raises(ValueError):
            append_equivalence(
                from_rubric="v2", to_rubric="v3",
                axes=["a"], except_axes=["b"],
                reason="r",
                repo_root=tmp_path,
            )

    def test_load_registry_preserves_declaration_order(self, tmp_path):
        for i in range(5):
            append_equivalence(
                from_rubric=f"v{i}", to_rubric=f"v{i+1}",
                reason=f"r{i}",
                repo_root=tmp_path,
            )
        registry = load_registry(repo_root=tmp_path)
        assert [e.from_rubric for e in registry] == ["v0", "v1", "v2", "v3", "v4"]
