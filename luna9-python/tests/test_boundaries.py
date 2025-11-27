"""
Tests for Boundaries framework.
"""

import pytest
from luna9.initiative.boundaries import (
    Boundaries,
    BoundariesConfig,
    PermissionLevel,
    describe_permission_level
)


class TestBoundaries:
    """Test Boundaries dataclass."""

    def test_boundaries_creation(self):
        """Test creating boundaries."""
        boundaries = Boundaries(
            core_values={"collaboration": "Working together"},
            permission_level="offered",
            trust_context="Test context"
        )

        assert "collaboration" in boundaries.core_values
        assert boundaries.permission_level == "offered"
        assert boundaries.trust_context == "Test context"


class TestBoundariesConfig:
    """Test boundary configuration factories."""

    def test_default_luna9(self):
        """Test default Luna 9 boundaries."""
        boundaries = BoundariesConfig.default_luna9()

        assert len(boundaries.core_values) == 4
        assert "collaboration" in boundaries.core_values
        assert "consent" in boundaries.core_values
        assert "honesty" in boundaries.core_values
        assert "kindness" in boundaries.core_values
        assert boundaries.permission_level == "offered"
        assert "new collaboration" in boundaries.trust_context.lower()

    def test_default_values_content(self):
        """Test that default values have meaningful descriptions."""
        boundaries = BoundariesConfig.default_luna9()

        # Each value should have a description
        for value_name, description in boundaries.core_values.items():
            assert len(description) > 10  # Not empty
            assert isinstance(description, str)

        # Spot-check specific content
        assert "working together" in boundaries.core_values["collaboration"].lower()
        assert "checking in" in boundaries.core_values["consent"].lower()
        assert "truth" in boundaries.core_values["honesty"].lower()
        assert "care" in boundaries.core_values["kindness"].lower()

    def test_minimal(self):
        """Test minimal boundaries."""
        boundaries = BoundariesConfig.minimal()

        assert len(boundaries.core_values) == 2
        assert "collaboration" in boundaries.core_values
        assert "honesty" in boundaries.core_values
        assert boundaries.permission_level == "offered"
        assert boundaries.trust_context == "Testing mode"

    def test_custom(self):
        """Test custom boundaries."""
        custom_values = {
            "transparency": "Full visibility into decisions",
            "respect": "Mutual regard for autonomy"
        }

        boundaries = BoundariesConfig.custom(
            values=custom_values,
            permission_level="welcomed",
            trust_context="Established partnership"
        )

        assert boundaries.core_values == custom_values
        assert boundaries.permission_level == "welcomed"
        assert boundaries.trust_context == "Established partnership"

    def test_custom_defaults(self):
        """Test custom boundaries with default arguments."""
        custom_values = {"test_value": "test description"}
        boundaries = BoundariesConfig.custom(values=custom_values)

        assert boundaries.permission_level == "offered"
        assert boundaries.trust_context == "Custom collaboration"


class TestPermissionLevel:
    """Test permission level constants and descriptions."""

    def test_permission_constants(self):
        """Test that permission level constants exist."""
        assert PermissionLevel.WELCOMED == "welcomed"
        assert PermissionLevel.OFFERED == "offered"
        assert PermissionLevel.ASKED == "asked"

    def test_describe_welcomed(self):
        """Test description for welcomed permission level."""
        desc = describe_permission_level(PermissionLevel.WELCOMED)

        assert "high autonomy" in desc.lower()
        assert "freely" in desc.lower()

    def test_describe_offered(self):
        """Test description for offered permission level."""
        desc = describe_permission_level(PermissionLevel.OFFERED)

        assert "medium autonomy" in desc.lower()
        assert "suggest" in desc.lower()

    def test_describe_asked(self):
        """Test description for asked permission level."""
        desc = describe_permission_level(PermissionLevel.ASKED)

        assert "low autonomy" in desc.lower()
        assert "explicit" in desc.lower()

    def test_describe_unknown(self):
        """Test description for unknown permission level."""
        desc = describe_permission_level("invalid_level")

        assert "unknown" in desc.lower()


class TestBoundariesIntegration:
    """Integration tests for boundaries."""

    def test_luna9_boundaries_completeness(self):
        """Test that Luna 9 boundaries are complete and well-formed."""
        boundaries = BoundariesConfig.default_luna9()

        # Should have all four core values
        required_values = ["collaboration", "consent", "honesty", "kindness"]
        for value in required_values:
            assert value in boundaries.core_values
            assert len(boundaries.core_values[value]) > 20  # Meaningful description

        # Permission level should be valid
        assert boundaries.permission_level in [
            PermissionLevel.WELCOMED,
            PermissionLevel.OFFERED,
            PermissionLevel.ASKED
        ]

        # Trust context should be set
        assert len(boundaries.trust_context) > 0

    def test_boundaries_immutable_pattern(self):
        """Test that boundaries follow immutable dataclass pattern."""
        boundaries = BoundariesConfig.default_luna9()

        # Can access attributes
        assert boundaries.permission_level == "offered"

        # Dataclass allows modification (not frozen), but best practice is to create new instances
        # This test documents the current behavior
        original_level = boundaries.permission_level
        boundaries.permission_level = "welcomed"
        assert boundaries.permission_level == "welcomed"

        # Reset for other tests
        boundaries.permission_level = original_level

    def test_multiple_boundary_sets(self):
        """Test creating multiple independent boundary sets."""
        boundaries1 = BoundariesConfig.default_luna9()
        boundaries2 = BoundariesConfig.minimal()
        boundaries3 = BoundariesConfig.custom({"test": "value"})

        # Each should be independent
        assert len(boundaries1.core_values) != len(boundaries2.core_values)
        assert boundaries1.core_values != boundaries3.core_values
        assert boundaries2.trust_context != boundaries3.trust_context
