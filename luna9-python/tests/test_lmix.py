"""
Tests for LMIX translation layer.
"""

import pytest
from luna9.initiative.lmix import LMIXLexicon, LMIXTranslator
from luna9.initiative.state import StateContext
from luna9.initiative.possibilities import Possibilities
from luna9.initiative.boundaries import Boundaries


class TestLMIXLexicon:
    """Test LMIX vocabulary mappings."""

    def test_curvature_mapping(self):
        """Test curvature values map to correct terms."""
        assert LMIXLexicon.map_value(0.005, LMIXLexicon.CURVATURE) == "straightforward"
        assert LMIXLexicon.map_value(0.02, LMIXLexicon.CURVATURE) == "meeting point"
        assert LMIXLexicon.map_value(0.04, LMIXLexicon.CURVATURE) == "junction"
        assert LMIXLexicon.map_value(0.06, LMIXLexicon.CURVATURE) == "nexus"

    def test_distance_mapping(self):
        """Test distance values map to correct terms."""
        assert LMIXLexicon.map_value(0.05, LMIXLexicon.DISTANCE) == "immediate"
        assert LMIXLexicon.map_value(0.2, LMIXLexicon.DISTANCE) == "nearby"
        assert LMIXLexicon.map_value(0.4, LMIXLexicon.DISTANCE) == "related"
        assert LMIXLexicon.map_value(0.7, LMIXLexicon.DISTANCE) == "tangential"
        assert LMIXLexicon.map_value(0.9, LMIXLexicon.DISTANCE) == "distant"

    def test_similarity_mapping(self):
        """Test similarity values map to correct terms."""
        assert LMIXLexicon.map_value(0.9, LMIXLexicon.SIMILARITY) == "resonant"
        assert LMIXLexicon.map_value(0.6, LMIXLexicon.SIMILARITY) == "adjacent"
        assert LMIXLexicon.map_value(0.3, LMIXLexicon.SIMILARITY) == "contrasting"
        assert LMIXLexicon.map_value(0.1, LMIXLexicon.SIMILARITY) == "in tension"

    def test_tension_mapping(self):
        """Test tension values map to correct terms."""
        assert LMIXLexicon.map_value(0.05, LMIXLexicon.TENSION) == "settled"
        assert LMIXLexicon.map_value(0.2, LMIXLexicon.TENSION) == "quiet"
        assert LMIXLexicon.map_value(0.4, LMIXLexicon.TENSION) == "open"
        assert LMIXLexicon.map_value(0.7, LMIXLexicon.TENSION) == "pulling"
        assert LMIXLexicon.map_value(0.9, LMIXLexicon.TENSION) == "urgent"

    def test_trajectory_mapping(self):
        """Test trajectory keys map to correct terms."""
        assert LMIXLexicon.map_key("linear", LMIXLexicon.TRAJECTORY) == "threading"
        assert LMIXLexicon.map_key("spiral", LMIXLexicon.TRAJECTORY) == "deepening"
        assert LMIXLexicon.map_key("jumping", LMIXLexicon.TRAJECTORY) == "leaping"
        assert LMIXLexicon.map_key("stationary", LMIXLexicon.TRAJECTORY) == "dwelling"

    def test_exploration_mapping(self):
        """Test exploration status keys map to correct terms."""
        assert LMIXLexicon.map_key("well_visited", LMIXLexicon.EXPLORATION) == "familiar"
        assert LMIXLexicon.map_key("visited", LMIXLexicon.EXPLORATION) == "known"
        assert LMIXLexicon.map_key("touched", LMIXLexicon.EXPLORATION) == "glimpsed"
        assert LMIXLexicon.map_key("unvisited", LMIXLexicon.EXPLORATION) == "unexplored"
        assert LMIXLexicon.map_key("new", LMIXLexicon.EXPLORATION) == "new"

    def test_path_curvature_mapping(self):
        """Test path curvature values map to correct terms."""
        assert LMIXLexicon.map_value(0.01, LMIXLexicon.PATH_CURVATURE) == "flowing"
        assert LMIXLexicon.map_value(0.03, LMIXLexicon.PATH_CURVATURE) == "bridging"
        assert LMIXLexicon.map_value(0.05, LMIXLexicon.PATH_CURVATURE) == "shifting"


class TestLMIXTranslator:
    """Test LMIX prose generation."""

    @pytest.fixture
    def translator(self):
        """Create translator instance."""
        return LMIXTranslator()

    def test_render_curvature(self, translator):
        """Test curvature rendering."""
        assert translator.render_curvature(0.005) == "straightforward"
        assert translator.render_curvature(0.04) == "junction"

    def test_render_distance(self, translator):
        """Test distance rendering."""
        assert translator.render_distance(0.05) == "immediate"
        assert translator.render_distance(0.9) == "distant"

    def test_render_trajectory(self, translator):
        """Test trajectory rendering."""
        assert translator.render_trajectory("linear") == "threading"
        assert translator.render_trajectory("jumping") == "leaping"

    def test_render_state_basic(self, translator):
        """Test basic state rendering."""
        state = StateContext(
            position=(0.5, 0.5),
            domain_path="foundation/test",
            ambient_memories=[],
            trajectory="linear"
        )
        prose = translator.render_state(state)

        assert "0.50, 0.50" in prose
        assert "foundation/test" in prose
        assert "threading" in prose

    def test_render_state_with_memories(self, translator):
        """Test state rendering with ambient memories."""
        state = StateContext(
            position=(0.3, 0.7),
            domain_path="project/luna9",
            ambient_memories=[
                {"distance": 0.05, "text": "First memory about initiative"},
                {"distance": 0.2, "text": "Second memory about LMIX"},
            ],
            trajectory="jumping"
        )
        prose = translator.render_state(state)

        assert "immediate" in prose
        assert "First memory" in prose
        assert "nearby" in prose
        assert "leaping" in prose

    def test_render_possibilities_empty(self, translator):
        """Test possibilities rendering when nothing present."""
        possibilities = Possibilities(
            high_curvature_regions=[],
            tension_threads=[],
            unexplored_nearby=[]
        )
        prose = translator.render_possibilities(possibilities)

        assert "quiet" in prose.lower()

    def test_render_possibilities_with_content(self, translator):
        """Test possibilities rendering with content."""
        possibilities = Possibilities(
            high_curvature_regions=[
                {"curvature": 0.04, "description": "consent and boundaries converge"}
            ],
            tension_threads=[
                {"energy": 0.7, "query": "How does initiative emerge?"}
            ],
            unexplored_nearby=[
                {"distance": 0.2, "region": "geometric security applications"}
            ]
        )
        prose = translator.render_possibilities(possibilities)

        assert "junction" in prose
        assert "consent and boundaries" in prose
        assert "pulling" in prose
        assert "How does initiative emerge" in prose
        assert "nearby" in prose
        assert "geometric security" in prose

    def test_render_boundaries(self, translator):
        """Test boundaries rendering."""
        boundaries = Boundaries(
            core_values={
                "collaboration": "Working together toward shared goals",
                "consent": "Checking in before significant changes"
            },
            permission_level="offered",
            trust_context="New collaboration, discovery mode"
        )
        prose = translator.render_boundaries(boundaries)

        assert "Collaboration" in prose
        assert "Working together" in prose
        assert "offered" in prose
        assert "Suggest first" in prose
        assert "New collaboration" in prose

    def test_render_full_context(self, translator):
        """Test full context rendering."""
        state = StateContext(
            position=(0.5, 0.5),
            domain_path="test/domain",
            ambient_memories=[],
            trajectory="linear"
        )
        possibilities = Possibilities(
            high_curvature_regions=[],
            tension_threads=[],
            unexplored_nearby=[]
        )
        boundaries = Boundaries(
            core_values={"collaboration": "Working together"},
            permission_level="offered",
            trust_context="Test context"
        )

        prose = translator.render_full_context(state, possibilities, boundaries)

        assert "# Initiative Context" in prose
        assert "## Current State" in prose
        assert "## Possibilities" in prose
        assert "## Boundaries" in prose

    def test_determinism(self, translator):
        """Test that same input produces same output (deterministic)."""
        state = StateContext(
            position=(0.3, 0.7),
            domain_path="test",
            ambient_memories=[{"distance": 0.1, "text": "test"}],
            trajectory="jumping"
        )

        prose1 = translator.render_state(state)
        prose2 = translator.render_state(state)

        assert prose1 == prose2

    def test_prose_format(self, translator):
        """Test that prose is readable and well-formatted."""
        state = StateContext(
            position=(0.5, 0.5),
            domain_path="test",
            ambient_memories=[],
            trajectory="linear"
        )
        prose = translator.render_state(state)

        # Should have proper formatting
        assert len(prose) > 0
        assert prose[0].isupper() or prose[0] == "("  # Starts with capital or coordinate
        assert "\n" in prose  # Has line breaks for readability
