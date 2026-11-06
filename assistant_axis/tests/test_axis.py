"""
Tests for axis computation utilities.
"""

import tempfile
import pytest
import torch

from assistant_axis.axis import (
    compute_axis,
    project,
    project_batch,
    save_axis,
    load_axis,
    load_axis_with_metadata,
    load_role_vector,
    slot_labels,
    cosine_similarity_per_layer,
    axis_norm_per_layer,
)


class TestComputeAxis:
    """Tests for compute_axis function."""

    def test_basic_computation(self):
        """Axis should be default - role activations."""
        n_layers, hidden_dim = 4, 8
        role_acts = torch.randn(10, n_layers, hidden_dim)
        default_acts = torch.randn(10, n_layers, hidden_dim)

        axis = compute_axis(role_acts, default_acts)

        expected = default_acts.mean(dim=0) - role_acts.mean(dim=0)
        assert axis.shape == (n_layers, hidden_dim)
        assert torch.allclose(axis, expected)

    def test_output_shape(self):
        """Output shape should be (n_layers, hidden_dim)."""
        n_layers, hidden_dim = 32, 4096
        role_acts = torch.randn(50, n_layers, hidden_dim)
        default_acts = torch.randn(20, n_layers, hidden_dim)

        axis = compute_axis(role_acts, default_acts)

        assert axis.shape == (n_layers, hidden_dim)

    def test_different_sample_counts(self):
        """Should work with different numbers of samples."""
        n_layers, hidden_dim = 4, 16
        role_acts = torch.randn(100, n_layers, hidden_dim)
        default_acts = torch.randn(5, n_layers, hidden_dim)

        axis = compute_axis(role_acts, default_acts)

        assert axis.shape == (n_layers, hidden_dim)


class TestProject:
    """Tests for project function."""

    def test_basic_projection(self):
        """Projection should be dot product with normalized axis."""
        n_layers, hidden_dim = 4, 8
        activations = torch.randn(n_layers, hidden_dim)
        axis = torch.randn(n_layers, hidden_dim)
        layer = 2

        result = project(activations, axis, layer)

        # Manual calculation
        act = activations[layer].float()
        ax = axis[layer].float()
        ax_normalized = ax / ax.norm()
        expected = float(act @ ax_normalized)

        assert abs(result - expected) < 1e-5

    def test_unnormalized(self):
        """With normalize=False, should be raw dot product."""
        n_layers, hidden_dim = 4, 8
        activations = torch.randn(n_layers, hidden_dim)
        axis = torch.randn(n_layers, hidden_dim)
        layer = 1

        result = project(activations, axis, layer, normalize=False)

        expected = float(activations[layer].float() @ axis[layer].float())
        assert abs(result - expected) < 1e-5

    def test_returns_float(self):
        """Should return a Python float."""
        activations = torch.randn(4, 8)
        axis = torch.randn(4, 8)

        result = project(activations, axis, layer=0)

        assert isinstance(result, float)


class TestProjectBatch:
    """Tests for project_batch function."""

    def test_batch_projection(self):
        """Should project multiple activations."""
        n_samples, n_layers, hidden_dim = 10, 4, 8
        activations = torch.randn(n_samples, n_layers, hidden_dim)
        axis = torch.randn(n_layers, hidden_dim)
        layer = 2

        results = project_batch(activations, axis, layer)

        assert len(results) == n_samples

        # Verify each result matches individual projection
        for i, result in enumerate(results):
            expected = project(activations[i], axis, layer)
            assert abs(result - expected) < 1e-5


class TestSaveLoadAxis:
    """Tests for save_axis and load_axis functions."""

    def test_save_and_load(self):
        """Saved axis should load correctly."""
        axis = torch.randn(32, 4096)

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            save_axis(axis, f.name)
            loaded = load_axis(f.name)

        assert torch.allclose(axis, loaded)

    def test_preserves_dtype(self):
        """Should preserve tensor dtype."""
        axis = torch.randn(4, 8, dtype=torch.float32)

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            save_axis(axis, f.name)
            loaded = load_axis(f.name)

        assert loaded.dtype == axis.dtype


class TestSlottedFiles:
    """The (n_slots, n_layers, hidden) layout steps 2, 4 and 5 write."""

    META = {
        "header_tokens": ["<|im_start|>", "assistant", "\n", "<think>", "\n\n",
                          "</think>", "\n\n"],
        "header_ids": [151644, 77091, 198, 151667, 271, 151668, 271],
        "model_name": "Qwen/Qwen3-32B",
    }

    def test_slot_labels_body_mean_first_then_header_tokens_in_order(self):
        labels = slot_labels(self.META)
        assert labels[0] == "body-mean"
        assert labels[1:] == self.META["header_tokens"]
        assert len(labels) == 8          # one per slot of an 8-slot axis

    def test_slot_labels_without_metadata(self):
        assert slot_labels(None) == ["body-mean"]
        assert slot_labels({}) == ["body-mean"]

    def test_save_axis_is_atomic_and_round_trips_metadata(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TMPDIR", str(tmp_path))
        axis = torch.randn(8, 4, 16, dtype=torch.bfloat16)
        path = tmp_path / "sub" / "axis.pt"
        save_axis(axis, str(path), metadata=self.META)
        loaded, meta = load_axis_with_metadata(str(path))
        assert torch.equal(loaded, axis) and loaded.dtype == torch.bfloat16
        assert meta == self.META
        assert len(slot_labels(meta)) == loaded.shape[0]
        assert not (tmp_path / "sub" / "axis.pt.inprogress").exists()

    def test_load_role_vector_dict_and_bare(self, tmp_path):
        v = torch.randn(8, 4, 16)
        torch.save({"vector": v, "type": "pos_3", "role": "pirate",
                    "metadata": self.META}, tmp_path / "pirate.pt")
        got, meta = load_role_vector(str(tmp_path / "pirate.pt"))
        assert torch.equal(got, v) and meta == self.META
        torch.save(v, tmp_path / "bare.pt")
        got, meta = load_role_vector(str(tmp_path / "bare.pt"))
        assert torch.equal(got, v) and meta == {}
        torch.save({"axis": v}, tmp_path / "wrong.pt")
        with pytest.raises(ValueError, match="vector"):
            load_role_vector(str(tmp_path / "wrong.pt"))

    def test_project_takes_the_slot_first(self):
        """project() indexes the layer of a 2-D axis: for a slotted axis the
        caller passes axis[slot] (steering reads axis[slot, layer])."""
        axis = torch.randn(8, 4, 16)
        act = torch.randn(4, 16)
        slot, layer = 3, 2
        expected = float(act[layer] @ (axis[slot, layer] / axis[slot, layer].norm()))
        assert abs(project(act, axis[slot], layer) - expected) < 1e-4


class TestCosineSimilarityPerLayer:
    """Tests for cosine_similarity_per_layer function."""

    def test_identical_vectors(self):
        """Identical vectors should have similarity 1."""
        v = torch.randn(4, 8)

        similarities = cosine_similarity_per_layer(v, v)

        assert len(similarities) == 4
        for sim in similarities:
            assert abs(sim - 1.0) < 1e-5

    def test_opposite_vectors(self):
        """Opposite vectors should have similarity -1."""
        v = torch.randn(4, 8)

        similarities = cosine_similarity_per_layer(v, -v)

        for sim in similarities:
            assert abs(sim + 1.0) < 1e-5

    def test_orthogonal_vectors(self):
        """Orthogonal vectors should have similarity ~0."""
        # Create orthogonal vectors
        v1 = torch.zeros(1, 4)
        v1[0, 0] = 1.0
        v2 = torch.zeros(1, 4)
        v2[0, 1] = 1.0

        similarities = cosine_similarity_per_layer(v1, v2)

        assert abs(similarities[0]) < 1e-5


class TestAxisNormPerLayer:
    """Tests for axis_norm_per_layer function."""

    def test_returns_correct_length(self):
        """Should return norm for each layer."""
        axis = torch.randn(32, 4096)

        norms = axis_norm_per_layer(axis)

        assert len(norms) == 32

    def test_values_are_positive(self):
        """Norms should be positive."""
        axis = torch.randn(4, 8)

        norms = axis_norm_per_layer(axis)

        for norm in norms:
            assert norm >= 0

    def test_zero_vector(self):
        """Zero vector should have norm 0."""
        axis = torch.zeros(2, 4)

        norms = axis_norm_per_layer(axis)

        for norm in norms:
            assert norm == 0


class TestRequireHeaderLayout:
    """The slot-layout guard of the multi-model plan (§ 5.6, § 9 test 10):
    the scripts whose default slot or slot names are Qwen3's refuse another
    layout.  Each script's call is reviewed in its diff, not tested here."""

    SEVEN = ["<|im_start|>", "assistant", "\n", "<think>", "\n\n", "</think>", "\n\n"]

    def test_passes_on_the_seven(self):
        from assistant_axis.axis import require_header_layout
        assert require_header_layout({"header_tokens": list(self.SEVEN)}, where="x.py") is None

    def test_raises_on_another_header_naming_the_script(self):
        from assistant_axis.axis import SlotLayoutError, require_header_layout
        with pytest.raises(SlotLayoutError) as e:
            require_header_layout({"header_tokens": ["a", "b"]}, where="whitening_k_sweep.py")
        msg = str(e.value)
        assert msg.startswith("whitening_k_sweep.py: ")
        assert "['a', 'b']" in msg and "</think>" in msg and "pass --slot explicitly" in msg

    def test_raises_on_metadata_without_a_header(self):
        from assistant_axis.axis import SlotLayoutError, require_header_layout
        for meta in (None, {}, {"extract_headers": False}):
            with pytest.raises(SlotLayoutError):
                require_header_layout(meta, where="w")

    def test_an_expected_layout_can_be_given(self):
        from assistant_axis.axis import require_header_layout
        require_header_layout({"header_tokens": ["a", "b"]}, expected=("a", "b"))


class TestTreeHeaderLayout:
    """The tree-level form of the guard: the metadata of one vector of a
    data directory (what the consumer scripts call)."""

    SEVEN = ["<|im_start|>", "assistant", "\n", "<think>", "\n\n", "</think>", "\n\n"]

    @staticmethod
    def _tree(tmp_path, header, sub="traits/vectors"):
        import torch
        d = tmp_path / "tree" / sub
        d.mkdir(parents=True)
        meta = {"header_tokens": header} if header is not None else {}
        torch.save({"vector": torch.zeros(1 + len(header or []), 2, 3), "metadata": meta},
                   d / "a.pt")
        return tmp_path / "tree"

    def test_a_qwen3_tree_passes_and_returns_its_metadata(self, tmp_path):
        from assistant_axis.axis import is_qwen3_header, require_tree_header_layout
        meta = require_tree_header_layout(self._tree(tmp_path, list(self.SEVEN)), where="s.py")
        assert is_qwen3_header(meta)

    def test_another_tree_raises_with_the_hint(self, tmp_path):
        from assistant_axis.axis import SlotLayoutError, require_tree_header_layout
        tree = self._tree(tmp_path, ["<start_of_turn>", "model", "\n"], sub="roles/vectors")
        with pytest.raises(SlotLayoutError, match="this script's names are Qwen3's"):
            require_tree_header_layout(tree, where="s.py",
                                       hint="this script's names are Qwen3's")

    def test_a_tree_without_a_readable_vector_is_not_checked(self, tmp_path, caplog):
        # No vector, or only zero-byte placeholders (never a vector: skipped,
        # not retried with the NFS backoff): the script's own checks report
        # the missing tree; the guard logs that it could not check.
        import logging
        from assistant_axis.axis import require_tree_header_layout, tree_vector_metadata
        (tmp_path / "empty" / "traits" / "vectors").mkdir(parents=True)
        (tmp_path / "empty" / "traits" / "vectors" / "a.pt").write_bytes(b"")
        assert tree_vector_metadata(tmp_path / "empty") is None
        with caplog.at_level(logging.WARNING):
            assert require_tree_header_layout(tmp_path / "empty", where="s.py") == {}
        assert "slot layout not checked" in caplog.text

    def test_a_readable_vector_without_header_metadata_raises(self, tmp_path):
        from assistant_axis.axis import SlotLayoutError, require_tree_header_layout
        with pytest.raises(SlotLayoutError):
            require_tree_header_layout(self._tree(tmp_path, None), where="s.py")

    def test_the_8slot_tree_passes(self):
        # Read only; skipped where the tree is not present.
        from pathlib import Path
        from assistant_axis.axis import require_tree_header_layout
        tree = Path(__file__).resolve().parents[2] / "runpod_workspace" / "qwen" / \
            "qwen-3-32b Roger 8slot"
        if not (tree / "traits" / "vectors").is_dir():
            pytest.skip("no 8slot tree here")
        meta = require_tree_header_layout(tree, where="test")
        assert meta["model_name"] == "Qwen/Qwen3-32B"
