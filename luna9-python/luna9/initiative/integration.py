"""
Initiative Engine

Orchestrates State + Possibilities + Boundaries → LMIX → LLM.

The hypothesis: When an LLM receives geometric context rendered as natural
language, initiative emerges naturally without explicit "when to act" logic.
"""

from typing import Optional, Dict, Any, List
from dataclasses import dataclass

from luna9.components.domain import Domain
from luna9.initiative.state import StateSurface
from luna9.initiative.possibilities import PossibilitiesMapper
from luna9.initiative.boundaries import (
    Boundaries,
    BoundariesConfig,
    ChoirConfig,
    CognitionMode,
)
from luna9.initiative.lmix import LMIXTranslator
from luna9.initiative.dual_mode import DualModeRetriever


@dataclass
class InitiativeContext:
    """
    Complete context for initiative emergence.

    Contains State + Possibilities + Boundaries rendered as natural language.
    """
    state_prose: str           # Current position, ambient context, trajectory
    possibilities_prose: str   # High-curvature regions, tension threads
    boundaries_prose: str      # Values, permissions, trust context
    full_context: str          # Combined context ready for LLM
    cognition_mode: CognitionMode = CognitionMode.SINGLE  # Single or Choir


@dataclass
class ChoirResponse:
    """
    Response from choir mode (multiple models).

    Attributes:
        responses: Dict mapping role name to model response
        synthesis: Combined/synthesized response (if applicable)
        model_ids: Dict mapping role to actual model ID used
    """
    responses: Dict[str, str]  # role -> response
    synthesis: Optional[str] = None  # Combined response
    model_ids: Optional[Dict[str, str]] = None  # role -> model_id


class InitiativeEngine:
    """
    Orchestrates initiative architecture components.

    Surfaces State, computes Possibilities, applies Boundaries, and renders
    everything via LMIX for LLM consumption.
    """

    def __init__(
        self,
        domain: Domain,
        boundaries: Optional[Boundaries] = None,
        choir_config: Optional[ChoirConfig] = None,
        client: Optional[Any] = None
    ):
        """
        Initialize Initiative engine.

        Args:
            domain: Luna 9 domain (provides memory surface)
            boundaries: Boundary configuration (defaults to Luna 9 defaults)
            choir_config: Choir configuration (None = single model mode)
            client: OpenRouter client (required if using choir mode)

        Raises:
            ValueError: If choir enabled but no client provided
        """
        self.domain = domain
        self.boundaries = boundaries or BoundariesConfig.default_luna9()
        self.choir_config = choir_config
        self.client = client

        # Validate choir configuration
        if choir_config and choir_config.enabled:
            if client is None:
                raise ValueError(
                    "Choir mode enabled but no OpenRouter client provided. "
                    "Pass client parameter to InitiativeEngine."
                )

        # Initialize components
        self.state_surface = StateSurface(domain)
        self.possibilities_mapper = PossibilitiesMapper(domain)
        self.lmix = LMIXTranslator()

        # Initialize dual-mode retriever (lazy - only if dual_mode is used)
        self._dual_mode_retriever: Optional[DualModeRetriever] = None

    def surface_initiative_context(
        self,
        query: Optional[str] = None,
        top_k: int = 5,
        k_retrieve_per_region: int = 3,
        max_ambient_chars: int = 200,
        max_region_messages: int = 2,
        max_message_chars: int = 150,
        use_dual_mode: bool = False,
        use_sigmoid: bool = False,
        dual_mode_params: Optional[Dict[str, Any]] = None
    ) -> InitiativeContext:
        """
        Surface complete initiative context.

        NOW WITH DUAL-MODE: Can use both surface navigation and flow suppression.

        Args:
            query: Optional query to ground position (if None, uses last position)
            top_k: Number of possibilities to surface
            k_retrieve_per_region: Messages to retrieve per curvature region
            max_ambient_chars: Max chars for ambient memories
            max_region_messages: Max messages per curvature region
            max_message_chars: Max chars per message
            use_dual_mode: Enable dual-mode retrieval (surface + suppression)
            use_sigmoid: Use sigmoid suppression instead of power law
            dual_mode_params: Optional params for dual-mode (surface_weight, etc.)

        Returns:
            Initiative context with all components rendered as prose
        """
        # Dual-mode path: Use DualModeRetriever instead of State + Possibilities
        if use_dual_mode:
            # Lazy init dual-mode retriever
            if self._dual_mode_retriever is None:
                params = dual_mode_params or {}
                self._dual_mode_retriever = DualModeRetriever(
                    domain=self.domain,
                    use_sigmoid=use_sigmoid,
                    **params
                )

            # Retrieve using dual-mode
            dual_context = self._dual_mode_retriever.retrieve(
                query=query or "",
                top_k=top_k,
                mode='dual'
            )

            # Render with LMIX
            full_context = self.lmix.render_dual_mode(
                dual_context,
                max_surface_chunks=top_k,
                max_suppression_chunks=top_k,
                max_chars=max_message_chars
            )

            # For dual-mode, we don't have separate state/possibilities prose
            # Return full_context in all fields for compatibility
            return InitiativeContext(
                state_prose="",
                possibilities_prose="",
                boundaries_prose="",
                full_context=full_context
            )

        # Standard path: Surface navigation only
        # 1. Surface State (already retrieves full content ✓)
        state = self.state_surface.get_current_state(query, k_ambient=top_k)
        state_prose = self.lmix.render_state(state, max_ambient_chars=max_ambient_chars)

        # 2. Compute Possibilities (NOW retrieves full content!)
        possibilities = self.possibilities_mapper.compute_possibilities(
            current_position=state.position,
            top_k=top_k,
            k_retrieve_per_region=k_retrieve_per_region
        )
        possibilities_prose = self.lmix.render_possibilities(
            possibilities,
            max_region_messages=max_region_messages,
            max_message_chars=max_message_chars
        )

        # 3. Render Boundaries
        boundaries_prose = self.lmix.render_boundaries(self.boundaries)

        # 4. Compose full context
        full_context = self._compose_context(
            state_prose,
            possibilities_prose,
            boundaries_prose
        )

        return InitiativeContext(
            state_prose=state_prose,
            possibilities_prose=possibilities_prose,
            boundaries_prose=boundaries_prose,
            full_context=full_context
        )

    def _compose_context(
        self,
        state: str,
        possibilities: str,
        boundaries: str
    ) -> str:
        """
        Compose complete context from components.

        Args:
            state: State prose
            possibilities: Possibilities prose
            boundaries: Boundaries prose

        Returns:
            Full context ready for LLM
        """
        sections = []

        # State section
        sections.append("=== WHERE YOU ARE ===\n")
        sections.append(state)
        sections.append("")

        # Possibilities section
        sections.append("=== WHAT'S PULLING ===\n")
        sections.append(possibilities)
        sections.append("")

        # Boundaries section
        sections.append("=== WHAT'S PERMITTED ===\n")
        sections.append(boundaries)
        sections.append("")

        return "\n".join(sections)

    def update_boundaries(self, new_boundaries: Boundaries) -> None:
        """
        Update boundary configuration.

        Args:
            new_boundaries: New boundary configuration
        """
        self.boundaries = new_boundaries

    def get_current_state_summary(self) -> Dict[str, Any]:
        """
        Get summary of current state (for debugging/inspection).

        Returns:
            Dict with position, trajectory, ambient context info

        Raises:
            ValueError: If no queries have been made yet
        """
        if not self.state_surface.query_history:
            raise ValueError("No state history - must call surface_initiative_context first")

        # Use last query from history
        last_query = self.state_surface.query_history[-1]
        state = self.state_surface.get_current_state(last_query, k_ambient=3)

        return {
            "position": {
                "u": state.position[0],
                "v": state.position[1]
            },
            "trajectory": state.trajectory,
            "ambient_memories_count": len(state.ambient_memories),
            "recent_queries_count": len(self.state_surface.query_history)
        }

    def reset_state(self) -> None:
        """
        Reset state tracking (clear history).

        Useful for starting fresh in testing scenarios.
        """
        self.state_surface = StateSurface(self.domain)

    # ========================================================================
    # CHOIR MODE - Multi-Model Ensemble Orchestration
    # ========================================================================

    def query_choir(
        self,
        context: InitiativeContext,
        user_query: Optional[str] = None
    ) -> ChoirResponse:
        """
        Query all models in choir with same initiative context.

        Args:
            context: Initiative context to present to all models
            user_query: Optional additional user query/prompt

        Returns:
            ChoirResponse with all model responses

        Raises:
            ValueError: If choir not configured or client missing
        """
        if not self.choir_config or not self.choir_config.enabled:
            raise ValueError("Choir mode not enabled")

        if not self.client:
            raise ValueError("No OpenRouter client configured")

        # Build prompt
        if user_query:
            prompt = f"{context.full_context}\n\n{user_query}"
        else:
            prompt = context.full_context

        # Query each model in choir
        responses = {}
        model_ids = {}

        for role, model_id in self.choir_config.models.items():
            try:
                response = self.client.complete_simple(prompt, model=model_id)
                responses[role] = response
                model_ids[role] = model_id
            except Exception as e:
                # Store error message as response
                responses[role] = f"[Error querying {model_id}: {str(e)}]"
                model_ids[role] = model_id

        # Synthesize responses
        synthesis = self._synthesize_responses(responses)

        return ChoirResponse(
            responses=responses,
            synthesis=synthesis,
            model_ids=model_ids
        )

    def _synthesize_responses(self, responses: Dict[str, str]) -> str:
        """
        Synthesize multiple choir responses.

        Currently implements CONCATENATE method only.

        Args:
            responses: Dict of role -> response

        Returns:
            Synthesized response string
        """
        if not responses:
            return ""

        # CONCATENATE: Present all responses (order-agnostic)
        sections = []
        sections.append("=== CHOIR RESPONSE ===\n")

        for role, response in responses.items():
            sections.append(f"--- {role.upper()} ---")
            sections.append(response)
            sections.append("")

        return "\n".join(sections)

    def query_with_initiative(
        self,
        query: str,
        top_k: int = 5,
        model: Optional[str] = None
    ) -> str:
        """
        Complete workflow: Surface context + query model(s).

        This is a convenience method that:
        1. Surfaces initiative context
        2. Queries choir (if enabled) or single model
        3. Returns response

        Args:
            query: User query
            top_k: Number of possibilities to surface
            model: Model ID (only used if choir disabled)

        Returns:
            Model response (single) or synthesized choir response

        Raises:
            ValueError: If single mode but no model specified and no client
        """
        # Surface context
        context = self.surface_initiative_context(query, top_k=top_k)

        # Determine mode
        if self.choir_config and self.choir_config.enabled:
            # Choir mode
            choir_response = self.query_choir(context, user_query=query)
            return choir_response.synthesis or ""
        else:
            # Single model mode
            if not self.client:
                raise ValueError(
                    "No OpenRouter client configured. "
                    "Pass client to InitiativeEngine or use surface_initiative_context() "
                    "to get context for external LLM."
                )

            if not model:
                # Try to get from choir config single model
                if self.choir_config and self.choir_config.models:
                    model = list(self.choir_config.models.values())[0]
                else:
                    raise ValueError("No model specified for single model mode")

            prompt = f"{context.full_context}\n\n{query}"
            return self.client.complete_simple(prompt, model=model)
