"""
Tests for Possibilities mapping component.
"""

import pytest
from unittest.mock import Mock, MagicMock
import numpy as np
from luna9.initiative.possibilities import Possibilities, PossibilitiesMapper


class TestPossibilities:
    """Test Possibilities dataclass."""

    def test_possibilities_creation(self):
        """Test creating possibilities."""
        possibilities = Possibilities(
            high_curvature_regions=[
                {"position": (0.5, 0.5), "curvature": 0.04}
            ],
            tension_threads=[
                {"query": "How does this work?", "energy": 0.7}
            ],
            unexplored_nearby=[
                {"region": "upper-left quadrant", "distance": 0.3}
            ]
        )

        assert len(possibilities.high_curvature_regions) == 1
        assert len(possibilities.tension_threads) == 1
        assert len(possibilities.unexplored_nearby) == 1


class TestPossibilitiesMapper:
    """Test possibilities mapping."""

    @pytest.fixture
    def mock_domain(self):
        """Create mock domain with surface."""
        domain = Mock()

        # Mock surface
        surface = Mock()
        surface.grid_m = 4
        surface.grid_n = 4
        surface.control_points = np.random.rand(4, 4, 384)
        surface.weights = np.ones((4, 4))
        surface.messages = [
            "First message about testing",
            "How does initiative emerge?",
            "Second statement",
            "What is the curvature?",
            "Third message"
        ]

        domain.surface = surface
        return domain

    def test_compute_possibilities_no_surface(self):
        """Test computing possibilities when domain has no surface."""
        domain = Mock()
        domain.surface = None

        mapper = PossibilitiesMapper(domain)
        possibilities = mapper.compute_possibilities((0.5, 0.5))

        assert len(possibilities.high_curvature_regions) == 0
        assert len(possibilities.tension_threads) == 0
        assert len(possibilities.unexplored_nearby) == 0

    def test_compute_possibilities_basic(self, mock_domain):
        """Test basic possibilities computation."""
        mapper = PossibilitiesMapper(mock_domain)
        possibilities = mapper.compute_possibilities((0.5, 0.5))

        # Should return some results (exact numbers depend on mocked data)
        assert isinstance(possibilities.high_curvature_regions, list)
        assert isinstance(possibilities.tension_threads, list)
        assert isinstance(possibilities.unexplored_nearby, list)

    def test_detect_tension(self, mock_domain):
        """Test tension detection finds questions."""
        mapper = PossibilitiesMapper(mock_domain)
        tension_threads = mapper._detect_tension(top_k=5)

        # Should find the two questions in our mock messages
        assert len(tension_threads) >= 2

        # Check that questions are identified
        queries = [t['query'] for t in tension_threads]
        assert any("initiative" in q.lower() for q in queries)
        assert any("curvature" in q.lower() for q in queries)

    def test_detect_tension_limit(self, mock_domain):
        """Test that tension detection respects top_k limit."""
        mapper = PossibilitiesMapper(mock_domain)
        tension_threads = mapper._detect_tension(top_k=1)

        assert len(tension_threads) <= 1

    def test_find_unexplored(self, mock_domain):
        """Test finding unexplored regions."""
        mapper = PossibilitiesMapper(mock_domain)

        # From center position
        unexplored = mapper._find_unexplored((0.5, 0.5), radius=1.0, top_k=5)

        # Should find some regions
        assert len(unexplored) > 0

        # Each should have required fields
        for region in unexplored:
            assert 'region' in region
            assert 'distance' in region
            assert 'position' in region

    def test_find_unexplored_radius(self, mock_domain):
        """Test that unexplored respects radius limit."""
        mapper = PossibilitiesMapper(mock_domain)

        # Small radius from corner
        unexplored = mapper._find_unexplored((0.1, 0.1), radius=0.3, top_k=10)

        # All should be within radius
        for region in unexplored:
            assert region['distance'] <= 0.3

    def test_find_unexplored_not_too_close(self, mock_domain):
        """Test that unexplored excludes very close regions."""
        mapper = PossibilitiesMapper(mock_domain)

        # From a candidate position
        unexplored = mapper._find_unexplored((0.9, 0.9), radius=1.0, top_k=10)

        # Should exclude regions too close (< 0.2)
        for region in unexplored:
            assert region['distance'] > 0.2

    def test_describe_curvature_region(self, mock_domain):
        """Test describing curvature regions."""
        mapper = PossibilitiesMapper(mock_domain)

        # Near a control point
        description = mapper._describe_curvature_region((0.5, 0.5))

        # Should return a string
        assert isinstance(description, str)
        assert len(description) > 0

    def test_invalidate_cache(self, mock_domain):
        """Test cache invalidation."""
        mapper = PossibilitiesMapper(mock_domain)

        # Set cache
        mapper._curvature_cache = np.array([1, 2, 3])

        # Invalidate
        mapper.invalidate_cache()

        assert mapper._curvature_cache is None

    def test_high_curvature_regions_sorted(self, mock_domain):
        """Test that high-curvature regions are sorted by curvature."""
        mapper = PossibilitiesMapper(mock_domain)

        regions = mapper._find_high_curvature_regions(
            (0.5, 0.5), radius=1.0, top_k=3
        )

        if len(regions) > 1:
            # Should be sorted highest first
            curvatures = [r['curvature'] for r in regions]
            assert curvatures == sorted(curvatures, reverse=True)

    def test_possibilities_top_k_respected(self, mock_domain):
        """Test that top_k limits are respected."""
        mapper = PossibilitiesMapper(mock_domain)

        possibilities = mapper.compute_possibilities(
            (0.5, 0.5), radius=1.0, top_k=2
        )

        # Each category should respect top_k=2
        assert len(possibilities.high_curvature_regions) <= 2
        assert len(possibilities.tension_threads) <= 2
        assert len(possibilities.unexplored_nearby) <= 2

    def test_empty_messages(self):
        """Test with domain that has no messages."""
        domain = Mock()
        surface = Mock()
        surface.grid_m = 2
        surface.grid_n = 2
        surface.control_points = np.random.rand(2, 2, 384)
        surface.weights = np.ones((2, 2))
        surface.messages = []

        domain.surface = surface

        mapper = PossibilitiesMapper(domain)
        possibilities = mapper.compute_possibilities((0.5, 0.5))

        # Should still work, just with no tension threads
        assert len(possibilities.tension_threads) == 0
