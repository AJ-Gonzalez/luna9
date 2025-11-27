"""
Tests for State surfacing component.
"""

import pytest
from unittest.mock import Mock, MagicMock
from luna9.initiative.state import StateContext, StateSurface
import numpy as np


class TestStateContext:
    """Test StateContext dataclass."""

    def test_state_context_creation(self):
        """Test creating a state context."""
        state = StateContext(
            position=(0.5, 0.5),
            domain_path="foundation/test",
            ambient_memories=[
                {"text": "Memory 1", "distance": 0.1},
                {"text": "Memory 2", "distance": 0.2}
            ],
            trajectory="linear"
        )

        assert state.position == (0.5, 0.5)
        assert state.domain_path == "foundation/test"
        assert len(state.ambient_memories) == 2
        assert state.trajectory == "linear"


class TestStateSurface:
    """Test StateSurface trajectory detection and state building."""

    @pytest.fixture
    def mock_domain(self):
        """Create mock domain with surface."""
        domain = Mock()
        domain.get_path.return_value = "test/domain"

        # Mock surface
        surface = Mock()
        surface._get_embedding_model = Mock()

        # Mock embedding model
        model = Mock()
        model.encode = Mock(return_value=np.array([[0.1] * 384]))
        surface._get_embedding_model.return_value = model

        # Mock projection
        surface.project_embedding = Mock(return_value=(0.5, 0.5))

        domain.surface = surface

        # Mock query results
        domain.query = Mock(return_value={
            'results': [
                {
                    'message': 'Test message 1',
                    'distance': 0.1,
                    'score': 0.9,
                    'metadata': {}
                },
                {
                    'message': 'Test message 2',
                    'distance': 0.2,
                    'score': 0.8,
                    'metadata': {}
                }
            ]
        })

        return domain

    def test_get_current_state(self, mock_domain):
        """Test building current state."""
        state_surface = StateSurface(mock_domain)
        state = state_surface.get_current_state("test query")

        assert state.position == (0.5, 0.5)
        assert state.domain_path == "test/domain"
        assert len(state.ambient_memories) == 2
        assert state.ambient_memories[0]['text'] == 'Test message 1'
        assert state.trajectory in ["linear", "jumping", "deepening", "stationary"]

    def test_trajectory_stationary(self, mock_domain):
        """Test trajectory detection - stationary."""
        state_surface = StateSurface(mock_domain)

        # Add positions that are very close together
        state_surface.position_history.extend([
            (0.5, 0.5),
            (0.51, 0.51),
            (0.52, 0.52)
        ])

        trajectory = state_surface._detect_trajectory((0.53, 0.53))
        assert trajectory == "stationary"

    def test_trajectory_linear(self, mock_domain):
        """Test trajectory detection - linear."""
        state_surface = StateSurface(mock_domain)

        # Add positions in a linear progression
        state_surface.position_history.extend([
            (0.1, 0.1),
            (0.2, 0.2),
            (0.3, 0.3)
        ])

        trajectory = state_surface._detect_trajectory((0.4, 0.4))
        assert trajectory == "linear"

    def test_trajectory_jumping(self, mock_domain):
        """Test trajectory detection - jumping."""
        state_surface = StateSurface(mock_domain)

        # Add positions with large jumps
        state_surface.position_history.extend([
            (0.1, 0.1),
            (0.8, 0.8),
            (0.2, 0.2)
        ])

        trajectory = state_surface._detect_trajectory((0.9, 0.9))
        assert trajectory == "jumping"

    def test_trajectory_deepening(self, mock_domain):
        """Test trajectory detection - deepening."""
        state_surface = StateSurface(mock_domain)

        # Add positions that circle back (spiral pattern)
        state_surface.position_history.extend([
            (0.5, 0.5),
            (0.6, 0.5),
            (0.6, 0.6),
            (0.5, 0.6)
        ])

        # Return close to start
        trajectory = state_surface._detect_trajectory((0.51, 0.51))
        assert trajectory == "deepening"

    def test_update_trajectory(self, mock_domain):
        """Test trajectory update."""
        state_surface = StateSurface(mock_domain)

        state_surface.update_trajectory("query 1", (0.1, 0.1))
        state_surface.update_trajectory("query 2", (0.2, 0.2))

        assert len(state_surface.position_history) == 2
        assert len(state_surface.query_history) == 2
        assert state_surface.position_history[0] == (0.1, 0.1)
        assert state_surface.query_history[1] == "query 2"

    def test_history_max_length(self, mock_domain):
        """Test that history respects max length."""
        state_surface = StateSurface(mock_domain)

        # Add more than 10 positions
        for i in range(15):
            state_surface.update_trajectory(f"query {i}", (i * 0.1, i * 0.1))

        # Should only keep last 10
        assert len(state_surface.position_history) == 10
        assert len(state_surface.query_history) == 10

    def test_clear_history(self, mock_domain):
        """Test clearing history."""
        state_surface = StateSurface(mock_domain)

        state_surface.update_trajectory("query 1", (0.1, 0.1))
        state_surface.update_trajectory("query 2", (0.2, 0.2))

        state_surface.clear_history()

        assert len(state_surface.position_history) == 0
        assert len(state_surface.query_history) == 0

    def test_get_history(self, mock_domain):
        """Test getting history."""
        state_surface = StateSurface(mock_domain)

        state_surface.update_trajectory("query 1", (0.1, 0.1))
        state_surface.update_trajectory("query 2", (0.2, 0.2))

        positions = state_surface.get_position_history()
        queries = state_surface.get_query_history()

        assert len(positions) == 2
        assert len(queries) == 2
        assert positions[0] == (0.1, 0.1)
        assert queries[1] == "query 2"

    def test_no_surface_error(self):
        """Test error when domain has no surface."""
        domain = Mock()
        domain.surface = None

        state_surface = StateSurface(domain)

        with pytest.raises(ValueError, match="Domain has no surface"):
            state_surface.get_current_state("test query")

    def test_ambient_memories_limit(self, mock_domain):
        """Test that ambient memories respect k_ambient limit."""
        # Mock query with many results
        mock_domain.query = Mock(return_value={
            'results': [
                {'message': f'Message {i}', 'distance': i * 0.1, 'score': 1.0 - i * 0.1, 'metadata': {}}
                for i in range(10)
            ]
        })

        state_surface = StateSurface(mock_domain)
        state = state_surface.get_current_state("test query", k_ambient=3)

        # Should only include 3 memories
        assert len(state.ambient_memories) == 3

    def test_state_includes_metadata(self, mock_domain):
        """Test that state includes message metadata."""
        mock_domain.query = Mock(return_value={
            'results': [{
                'message': 'Test message',
                'distance': 0.1,
                'score': 0.9,
                'metadata': {'author': 'Test Author', 'chapter': 1}
            }]
        })

        state_surface = StateSurface(mock_domain)
        state = state_surface.get_current_state("test query")

        assert state.ambient_memories[0]['metadata']['author'] == 'Test Author'
        assert state.ambient_memories[0]['metadata']['chapter'] == 1
