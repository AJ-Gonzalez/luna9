"""
Tests for semantic surface dual-mode retrieval.
"""

import numpy as np
import pytest
from semantic_surface import SemanticSurface, create_surface_from_conversation


# Test conversation (16 messages about AI safety)
TEST_MESSAGES = [
    "What are the main risks of AGI?",
    "The main risks include misalignment, rapid capability gain, and unintended consequences.",
    "But isn't misalignment solvable with RLHF?",
    "RLHF helps but has limitations - it optimizes for human feedback, not true alignment.",
    "What about using formal verification?",
    "Formal verification is promising but currently doesn't scale to large neural networks.",
    "So what's the path forward?",
    "Multiple approaches: interpretability research, robustness testing, and iterative deployment.",
    "That sounds expensive.",
    "True, but the cost of getting it wrong is potentially catastrophic.",
    "Fair point. What about AI regulation?",
    "Regulation can help, but needs to be technically informed and internationally coordinated.",
    "Isn't that unrealistic?",
    "It's challenging but necessary - we've seen international coordination on nuclear weapons.",
    "Good analogy.",
    "Though AI is harder to regulate due to dual-use nature and rapid development."
]


class TestSemanticSurface:
    """Test semantic surface creation and basic properties."""

    def test_surface_creation(self):
        """Test that surface can be created from messages."""
        # Use simulated embeddings for speed
        np.random.seed(42)
        embeddings = np.random.randn(16, 384)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        surface = SemanticSurface(TEST_MESSAGES, embeddings=embeddings)

        assert surface.control_points.shape == (4, 4, 384)
        assert len(surface.messages) == 16
        assert surface.embedding_dim == 384

    def test_provenance_mapping(self):
        """Test bidirectional provenance mappings."""
        np.random.seed(42)
        embeddings = np.random.randn(16, 384)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        surface = SemanticSurface(TEST_MESSAGES, embeddings=embeddings)

        # Check that mappings are bidirectional
        for msg_idx in range(16):
            i, j = surface.provenance['msg_to_cp'][msg_idx]
            assert surface.provenance['cp_to_msg'][(i, j)] == msg_idx

        # Check all 16 messages are mapped
        assert len(surface.provenance['cp_to_msg']) == 16
        assert len(surface.provenance['msg_to_cp']) == 16


class TestInfluenceComputation:
    """Test smooth retrieval via influence weights."""

    def test_influence_sums_to_one(self):
        """Influence weights should sum to 1."""
        np.random.seed(42)
        embeddings = np.random.randn(16, 384)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        surface = SemanticSurface(TEST_MESSAGES, embeddings=embeddings)

        # Test at various points
        test_points = [(0.2, 0.3), (0.5, 0.5), (0.8, 0.7)]

        for u, v in test_points:
            influence = surface.compute_influence(u, v)

            # Check sum
            total_weight = sum(weight for _, weight in influence)
            assert abs(total_weight - 1.0) < 1e-10, f"Weights sum to {total_weight}, not 1.0"

    def test_influence_sorted(self):
        """Influence weights should be sorted descending."""
        np.random.seed(42)
        embeddings = np.random.randn(16, 384)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        surface = SemanticSurface(TEST_MESSAGES, embeddings=embeddings)

        influence = surface.compute_influence(0.5, 0.5)

        # Check sorted
        weights = [weight for _, weight in influence]
        assert weights == sorted(weights, reverse=True), "Weights not sorted descending"

    def test_influence_all_messages(self):
        """Influence should include all 16 messages."""
        np.random.seed(42)
        embeddings = np.random.randn(16, 384)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        surface = SemanticSurface(TEST_MESSAGES, embeddings=embeddings)

        influence = surface.compute_influence(0.5, 0.5)

        assert len(influence) == 16, f"Expected 16 influences, got {len(influence)}"

        # Check all message indices present
        indices = [idx for idx, _ in influence]
        assert set(indices) == set(range(16)), "Not all messages represented"


class TestNearestControlPoints:
    """Test exact retrieval via nearest control points."""

    def test_nearest_returns_k(self):
        """Should return exactly k nearest control points."""
        np.random.seed(42)
        embeddings = np.random.randn(16, 384)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        surface = SemanticSurface(TEST_MESSAGES, embeddings=embeddings)

        for k in [1, 3, 5, 10]:
            nearest = surface.nearest_control_points(0.5, 0.5, k=k)
            assert len(nearest) == k, f"Expected {k} results, got {len(nearest)}"

    def test_nearest_sorted_by_distance(self):
        """Results should be sorted by distance ascending."""
        np.random.seed(42)
        embeddings = np.random.randn(16, 384)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        surface = SemanticSurface(TEST_MESSAGES, embeddings=embeddings)

        nearest = surface.nearest_control_points(0.5, 0.5, k=8)

        distances = [dist for _, _, _, dist in nearest]
        assert distances == sorted(distances), "Distances not sorted ascending"

    def test_corner_nearest_to_corner(self):
        """Corner query should have that corner as nearest control point."""
        np.random.seed(42)
        embeddings = np.random.randn(16, 384)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        surface = SemanticSurface(TEST_MESSAGES, embeddings=embeddings)

        # Query at corner (0, 0)
        nearest = surface.nearest_control_points(0.0, 0.0, k=4)

        # First result should be control point [0, 0]
        i, j, msg_idx, dist = nearest[0]
        assert (i, j) == (0, 0), f"Expected corner [0,0], got [{i},{j}]"
        assert dist < 0.01, f"Distance to corner should be ~0, got {dist}"


class TestDualRetrieval:
    """Test unified query interface returning both modes."""

    def test_query_returns_both_modes(self):
        """Query should return both influence and nearest control points."""
        np.random.seed(42)
        embeddings = np.random.randn(16, 384)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        surface = SemanticSurface(TEST_MESSAGES, embeddings=embeddings)

        result = surface.query("What about RLHF?", k=5)

        # Check has both retrieval modes
        assert len(result.influence) == 16, "Should have all influences"
        assert len(result.nearest_control_points) == 5, "Should have k=5 nearest"

        # Check UV coordinates
        assert 0 <= result.uv[0] <= 1, f"u={result.uv[0]} out of bounds"
        assert 0 <= result.uv[1] <= 1, f"v={result.uv[1]} out of bounds"

        # Check curvature
        K, H = result.curvature
        assert not np.isnan(K) and not np.isnan(H), "Curvature should be valid"

    def test_get_messages_smooth(self):
        """Test message retrieval in smooth mode."""
        np.random.seed(42)
        embeddings = np.random.randn(16, 384)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        surface = SemanticSurface(TEST_MESSAGES, embeddings=embeddings)

        result = surface.query("AI safety", k=3)
        messages = result.get_messages(TEST_MESSAGES, mode='smooth', k=3)

        assert 'interpretation' in messages
        assert len(messages['interpretation']['messages']) == 3
        assert len(messages['interpretation']['weights']) == 3
        assert 'sources' not in messages  # Only smooth mode

    def test_get_messages_exact(self):
        """Test message retrieval in exact mode."""
        np.random.seed(42)
        embeddings = np.random.randn(16, 384)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        surface = SemanticSurface(TEST_MESSAGES, embeddings=embeddings)

        result = surface.query("regulation", k=3)
        messages = result.get_messages(TEST_MESSAGES, mode='exact', k=3)

        assert 'sources' in messages
        assert len(messages['sources']['messages']) == 3
        assert len(messages['sources']['distances']) == 3
        assert 'interpretation' not in messages  # Only exact mode

    def test_get_messages_both(self):
        """Test message retrieval with both modes."""
        np.random.seed(42)
        embeddings = np.random.randn(16, 384)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        surface = SemanticSurface(TEST_MESSAGES, embeddings=embeddings)

        result = surface.query("alignment", k=4)
        messages = result.get_messages(TEST_MESSAGES, mode='both', k=4)

        assert 'interpretation' in messages
        assert 'sources' in messages
        assert len(messages['interpretation']['messages']) == 4
        assert len(messages['sources']['messages']) == 4


class TestConvenienceFunction:
    """Test convenience function for surface creation."""

    @pytest.mark.skip(reason="Requires downloading model, slow test")
    def test_create_from_conversation(self):
        """Test creating surface from conversation with real embeddings."""
        surface = create_surface_from_conversation(TEST_MESSAGES)

        assert surface.control_points.shape[0:2] == (4, 4)
        assert len(surface.messages) == 16


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
