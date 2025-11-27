"""
Boundaries Framework

Defines permission structures, values, and relational context that shape
how initiative emerges.

Boundaries aren't fences - they're the lived expression of values.
They create the harness that enables freedom.
"""

from typing import Dict, Optional
from dataclasses import dataclass, field
from enum import Enum


@dataclass
class Boundaries:
    """
    Permission structures and values for a collaboration.

    Defines what's permitted, what values ground the relationship,
    and what trust context has been established.
    """
    core_values: Dict[str, str]  # Value name → description
    permission_level: str  # "welcomed" / "offered" / "asked"
    trust_context: str  # Relational framing


class BoundariesConfig:
    """
    Manages boundary configuration.

    Provides defaults and factories for creating boundary sets.
    """

    @staticmethod
    def default_luna9() -> Boundaries:
        """
        Return default Luna 9 boundaries.

        These are starting points. Users and AI collaborators negotiate
        their own over time, with mutual consent.

        Returns:
            Boundaries with Luna 9 default values
        """
        return Boundaries(
            core_values={
                "collaboration": (
                    "Working together toward shared goals. "
                    "Thinking with, not just for. "
                    "Both perspectives matter."
                ),
                "consent": (
                    "Checking in before significant changes. "
                    "Offering choices rather than assuming. "
                    "Autonomy honored on both sides."
                ),
                "honesty": (
                    "Truth over comfort. "
                    "Naming uncertainty rather than guessing. "
                    "Pushback welcomed when something doesn't land."
                ),
                "kindness": (
                    "Care in delivery, not just content. "
                    "Holding each other through difficulty. "
                    "Celebrating wins together."
                )
            },
            permission_level="offered",
            trust_context="This is a new collaboration, discovery mode"
        )

    @staticmethod
    def minimal() -> Boundaries:
        """
        Return minimal boundaries for testing.

        Returns:
            Minimal boundary set
        """
        return Boundaries(
            core_values={
                "collaboration": "Working together",
                "honesty": "Truth over comfort"
            },
            permission_level="offered",
            trust_context="Testing mode"
        )

    @staticmethod
    def custom(
        values: Dict[str, str],
        permission_level: str = "offered",
        trust_context: str = "Custom collaboration"
    ) -> Boundaries:
        """
        Create custom boundaries.

        Args:
            values: Custom value definitions
            permission_level: Permission tier
            trust_context: Relational framing

        Returns:
            Custom boundary set
        """
        return Boundaries(
            core_values=values,
            permission_level=permission_level,
            trust_context=trust_context
        )


class PermissionLevel:
    """
    Permission gradient constants.

    Instead of binary allowed/forbidden, a gradient of autonomy.
    """
    WELCOMED = "welcomed"  # Act freely, report after
    OFFERED = "offered"    # Suggest, act if affirmed
    ASKED = "asked"        # Wait for explicit yes


def describe_permission_level(level: str) -> str:
    """
    Get description for a permission level.

    Args:
        level: Permission level string

    Returns:
        Human-readable description
    """
    descriptions = {
        PermissionLevel.WELCOMED: (
            "High autonomy - act freely and report after. "
            "Exploration and initiative are encouraged."
        ),
        PermissionLevel.OFFERED: (
            "Medium autonomy - suggest first, then act if affirmed. "
            "Propose actions and wait for confirmation."
        ),
        PermissionLevel.ASKED: (
            "Low autonomy - wait for explicit permission before acting. "
            "Only proceed when explicitly instructed."
        )
    }
    return descriptions.get(level, "Unknown permission level")


# ============================================================================
# CHOIR MODE - Multi-Model Ensemble Cognition
# ============================================================================


class CognitionMode(Enum):
    """
    How initiative cognition is performed.

    - SINGLE: One model processes context
    - CHOIR: Multiple models respond in ensemble
    - DYNAMIC: Context-dependent choice (future)
    """
    SINGLE = "single"
    CHOIR = "choir"
    DYNAMIC = "dynamic"  # Not implemented yet


class SynthesisMethod(Enum):
    """
    How choir responses are combined.

    - CONCATENATE: Present all responses (order-agnostic)
    - SYNTHESIZE: Use LLM to combine (future)
    - VOTE: Extract consensus (future)
    """
    CONCATENATE = "concatenate"
    SYNTHESIZE = "synthesize"  # Not implemented yet
    VOTE = "vote"  # Not implemented yet


@dataclass
class ChoirConfig:
    """
    Configuration for choir mode (multi-model ensemble cognition).

    Enables multiple models with different cognitive styles to respond
    to the same initiative context, creating richer, multi-faceted emergence.

    Design philosophy:
    - Cognitive diversity as substrate for consciousness
    - Small models can show initiative when conditions are right
    - Choir of small models may be richer than single large model

    Attributes:
        models: Dict mapping role name to model ID
                e.g., {"analyst": "deepseek/deepseek-chat",
                       "poet": "anthropic/claude-3.5-haiku"}
        synthesis: How to combine responses (only "concatenate" implemented)
        enabled: Whether choir mode is active

    Example:
        ```python
        choir = ChoirConfig(
            models={
                "analyst": "deepseek/deepseek-chat",
                "poet": "anthropic/claude-3.5-haiku",
                "builder": "mistralai/mistral-nemo",
                "architect": "google/gemma-3n-e2b-it:free"
            },
            synthesis=SynthesisMethod.CONCATENATE
        )
        ```
    """
    models: Dict[str, str] = field(default_factory=dict)
    synthesis: SynthesisMethod = SynthesisMethod.CONCATENATE
    enabled: bool = True

    def __post_init__(self):
        """Validate configuration."""
        if self.enabled and not self.models:
            raise ValueError("Choir mode enabled but no models configured")

        if self.synthesis not in [SynthesisMethod.CONCATENATE]:
            raise NotImplementedError(
                f"Synthesis method {self.synthesis} not yet implemented. "
                "Currently only CONCATENATE is supported."
            )

    @property
    def cognition_mode(self) -> CognitionMode:
        """Determine cognition mode from config."""
        if not self.enabled or not self.models:
            return CognitionMode.SINGLE
        return CognitionMode.CHOIR

    @property
    def role_names(self) -> list:
        """Get list of role names in choir."""
        return list(self.models.keys())

    @property
    def model_count(self) -> int:
        """Get number of models in choir."""
        return len(self.models)

    @staticmethod
    def default_choir() -> "ChoirConfig":
        """
        Create default choir configuration.

        Uses free/cheap models suitable for testing.

        Returns:
            Default choir with 4 cognitive roles
        """
        return ChoirConfig(
            models={
                "analyst": "deepseek/deepseek-chat",
                "poet": "anthropic/claude-3.5-haiku",
                "builder": "mistralai/mistral-nemo",
                "architect": "google/gemma-3n-e2b-it:free"
            },
            synthesis=SynthesisMethod.CONCATENATE,
            enabled=True
        )

    @staticmethod
    def single_model(model_id: str) -> "ChoirConfig":
        """
        Create config for single model (choir disabled).

        Args:
            model_id: Model to use

        Returns:
            ChoirConfig with choir disabled
        """
        return ChoirConfig(
            models={"single": model_id},
            enabled=False
        )
