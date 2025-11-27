"""
LMIX (Language Model Input eXperience)

Translates geometric data into natural language prose using deterministic vocabulary.
The geometric layer does the math, then speaks the results in words.
"""

from typing import Dict, List, Tuple, Any
from dataclasses import dataclass


class LMIXLexicon:
    """
    Deterministic vocabulary for geometric properties.

    Maps geometric values to natural language terms. Consistent mapping reduces
    ambiguity - if "junction" always means high-curvature convergence, that's a
    feature, not a bug.
    """

    # Curvature → Conceptual Density
    CURVATURE = [
        (0.0, 0.01, "straightforward"),
        (0.01, 0.03, "meeting point"),
        (0.03, 0.05, "junction"),
        (0.05, float('inf'), "nexus"),
    ]

    # Distance → Relevance/Proximity
    DISTANCE = [
        (0.0, 0.1, "immediate"),
        (0.1, 0.3, "nearby"),
        (0.3, 0.6, "related"),
        (0.6, 0.8, "tangential"),
        (0.8, float('inf'), "distant"),
    ]

    # Similarity → Alignment
    SIMILARITY = [
        (0.8, float('inf'), "resonant"),
        (0.5, 0.8, "adjacent"),
        (0.2, 0.5, "contrasting"),
        (-float('inf'), 0.2, "in tension"),
    ]

    # Tension → Energy State
    TENSION = [
        (0.0, 0.1, "settled"),
        (0.1, 0.3, "quiet"),
        (0.3, 0.6, "open"),
        (0.6, 0.8, "pulling"),
        (0.8, float('inf'), "urgent"),
    ]

    # Exploration Status → Familiarity
    EXPLORATION = {
        "well_visited": "familiar",
        "visited": "known",
        "touched": "glimpsed",
        "unvisited": "unexplored",
        "new": "new",
    }

    # Movement/Trajectory → Motion Quality
    TRAJECTORY = {
        "linear": "threading",
        "spiral": "deepening",
        "jumping": "leaping",
        "stationary": "dwelling",
    }

    # Path Curvature → Transition Complexity
    PATH_CURVATURE = [
        (0.0, 0.02, "flowing"),
        (0.02, 0.04, "bridging"),
        (0.04, float('inf'), "shifting"),
    ]

    @staticmethod
    def map_value(value: float, mapping: List[Tuple[float, float, str]]) -> str:
        """Map a numeric value to a term using range-based mapping."""
        for min_val, max_val, term in mapping:
            if min_val <= value < max_val:
                return term
        # Fallback to last term if out of range
        return mapping[-1][2]

    @staticmethod
    def map_key(key: str, mapping: Dict[str, str]) -> str:
        """Map a key to a term using dictionary mapping."""
        return mapping.get(key, key)


class LMIXTranslator:
    """
    Translates geometric data to natural language prose.

    Uses template-based composition with deterministic vocabulary to render
    State, Possibilities, and Boundaries as LMIX prose for LLM consumption.
    """

    def __init__(self):
        self.lexicon = LMIXLexicon()

    def render_curvature(self, value: float) -> str:
        """Translate curvature value to term."""
        return self.lexicon.map_value(value, self.lexicon.CURVATURE)

    def render_distance(self, value: float) -> str:
        """Translate distance value to term."""
        return self.lexicon.map_value(value, self.lexicon.DISTANCE)

    def render_similarity(self, value: float) -> str:
        """Translate similarity value to term."""
        return self.lexicon.map_value(value, self.lexicon.SIMILARITY)

    def render_tension(self, value: float) -> str:
        """Translate tension value to term."""
        return self.lexicon.map_value(value, self.lexicon.TENSION)

    def render_trajectory(self, key: str) -> str:
        """Translate trajectory key to term."""
        return self.lexicon.map_key(key, self.lexicon.TRAJECTORY)

    def render_exploration(self, key: str) -> str:
        """Translate exploration status to term."""
        return self.lexicon.map_key(key, self.lexicon.EXPLORATION)

    def render_path_curvature(self, value: float) -> str:
        """Translate path curvature to term."""
        return self.lexicon.map_value(value, self.lexicon.PATH_CURVATURE)

    def render_state(self, state: 'StateContext', max_ambient_chars: int = 200) -> str:
        """
        Render state context as natural language prose.

        Args:
            state: StateContext with position, ambient memories, trajectory
            max_ambient_chars: Max chars per ambient memory (default 200, was 80)

        Returns:
            Natural language description of current state
        """
        from luna9.initiative.state import StateContext

        u, v = state.position
        trajectory_term = self.render_trajectory(state.trajectory)

        prose = f"You are currently at position ({u:.2f}, {v:.2f}) on the semantic surface"
        prose += f" in domain '{state.domain_path}'.\n\n"

        prose += f"Your recent movement has been {trajectory_term}"
        if state.trajectory == "linear":
            prose += " - threading through a single topic.\n\n"
        elif state.trajectory == "jumping":
            prose += " - leaping between areas.\n\n"
        elif state.trajectory == "deepening":
            prose += " - deepening into a region.\n\n"
        else:
            prose += " - dwelling in one place.\n\n"

        if state.ambient_memories:
            prose += "Ambient context (what's present without searching):\n"
            for i, memory in enumerate(state.ambient_memories[:5], 1):
                distance = memory.get('distance', 0.0)
                distance_term = self.render_distance(distance)

                # NEW: Use configurable truncation instead of hardcoded 80
                full_text = memory.get('text', '')
                if len(full_text) > max_ambient_chars:
                    text = full_text[:max_ambient_chars] + "..."
                else:
                    text = full_text

                prose += f"{i}. [{distance_term}] {text}\n"
            prose += "\n"

        return prose

    def render_possibilities(
        self,
        possibilities: 'Possibilities',
        max_region_messages: int = 2,
        max_message_chars: int = 150
    ) -> str:
        """
        Render possibilities as natural language prose.

        NOW INCLUDES: Full retrieved content from curvature regions, not just previews.

        Args:
            possibilities: Possibilities with curvature regions, tension threads
            max_region_messages: Max messages to show per curvature region (default 2)
            max_message_chars: Max chars per message (default 150)

        Returns:
            Natural language description of what's pulling
        """
        from luna9.initiative.possibilities import Possibilities

        prose = ""

        # High-curvature regions WITH retrieved content (ENHANCED!)
        if possibilities.high_curvature_regions:
            prose += "There are regions of high conceptual density nearby:\n\n"
            for region in possibilities.high_curvature_regions[:3]:
                curvature = region.get('curvature', 0.0)
                curvature_term = self.render_curvature(curvature)
                semantic_label = region.get('description', 'Unknown region')

                prose += f"- A {curvature_term} around {semantic_label}\n"

                # NEW: Include retrieved content
                retrieved = region.get('retrieved_content', [])
                if retrieved:
                    prose += "  Content at this junction:\n"
                    for i, msg in enumerate(retrieved[:max_region_messages], 1):
                        text = msg.get('text', '')
                        if len(text) > max_message_chars:
                            text = text[:max_message_chars] + "..."

                        dist = msg.get('distance', 0.0)
                        dist_term = self.render_distance(dist)
                        prose += f"    {i}. [{dist_term}] {text}\n"
                    prose += "\n"
            prose += "\n"

        # Tension threads (unchanged)
        if possibilities.tension_threads:
            prose += "Active threads with tension:\n"
            for thread in possibilities.tension_threads[:3]:
                energy = thread.get('energy', 0.0)
                tension_term = self.render_tension(energy)
                query = thread.get('query', 'Unknown query')
                prose += f"- {query} ({tension_term})\n"
            prose += "\n"

        # Unexplored regions (unchanged)
        if possibilities.unexplored_nearby:
            prose += "Unexplored territory nearby:\n"
            for region in possibilities.unexplored_nearby[:3]:
                distance = region.get('distance', 0.0)
                distance_term = self.render_distance(distance)
                region_name = region.get('region', 'Unknown region')
                prose += f"- {region_name} ({distance_term})\n"
            prose += "\n"

        if not prose:
            prose = "The surrounding semantic space is quiet - no strong attractors nearby.\n\n"

        return prose

    def render_boundaries(self, boundaries: 'Boundaries') -> str:
        """
        Render boundaries as natural language prose.

        Args:
            boundaries: Boundaries with values, permissions, trust context

        Returns:
            Natural language description of boundaries
        """
        from luna9.initiative.boundaries import Boundaries

        prose = "This collaboration is grounded in:\n\n"

        for value_name, value_desc in boundaries.core_values.items():
            prose += f"**{value_name.title()}:** {value_desc}\n\n"

        prose += f"Permission level: {boundaries.permission_level}\n"
        if boundaries.permission_level == "welcomed":
            prose += "  -> You're encouraged to act freely and report after.\n\n"
        elif boundaries.permission_level == "offered":
            prose += "  -> Suggest first, then act if affirmed.\n\n"
        elif boundaries.permission_level == "asked":
            prose += "  -> Wait for explicit permission before acting.\n\n"

        prose += f"Trust context: {boundaries.trust_context}\n\n"

        return prose

    def render_full_context(
        self,
        state: 'StateContext',
        possibilities: 'Possibilities',
        boundaries: 'Boundaries'
    ) -> str:
        """
        Render complete initiative context as LMIX prose.

        Composes State + Possibilities + Boundaries into a coherent system prompt.

        Args:
            state: Current state context
            possibilities: What's pulling
            boundaries: Permission structures and values

        Returns:
            Complete LMIX prose for LLM system prompt
        """
        prose = "# Initiative Context\n\n"
        prose += "## Current State\n\n"
        prose += self.render_state(state)
        prose += "\n## Possibilities\n\n"
        prose += self.render_possibilities(possibilities)
        prose += "\n## Boundaries\n\n"
        prose += self.render_boundaries(boundaries)

        return prose

    def render_suppression_results(
        self,
        revealed_chunks: List[Dict],
        max_chunks: int = 5,
        max_chars: int = 500
    ) -> str:
        """
        Render flow suppression results as natural language prose.

        Highlights dispersed signals revealed by suppressing cohesive flow.

        Args:
            revealed_chunks: List of chunks from FlowSuppressor.get_revealed_chunks()
            max_chunks: Maximum chunks to render
            max_chars: Max chars per chunk

        Returns:
            Natural language description of revealed dispersed signals
        """
        if not revealed_chunks:
            return "No dispersed signals revealed.\n"

        prose = "Dispersed signals (revealed by suppressing cohesive flow):\n\n"
        prose += "These are scattered references that were previously drowned out by "
        prose += "dominant narrative flow. They represent characters, themes, or events "
        prose += "that appear across the text but don't form cohesive sequences.\n\n"

        for i, chunk in enumerate(revealed_chunks[:max_chunks], 1):
            content = chunk.get('content', '')
            if len(content) > max_chars:
                content = content[:max_chars] + "..."

            score = chunk.get('score', chunk.get('query_relevance', 0.0))
            revelation_strength = "strong" if score > 0.15 else "moderate"

            prose += f"{i}. [{revelation_strength} signal] {content}\n\n"

        return prose

    def render_dual_mode(
        self,
        context: 'DualModeContext',
        max_surface_chunks: int = 5,
        max_suppression_chunks: int = 5,
        max_chars: int = 500
    ) -> str:
        """
        Render dual-mode context as natural language prose.

        Combines cohesive flow (surface navigation) and dispersed signals
        (flow suppression) into a comprehensive narrative.

        Args:
            context: DualModeContext from DualModeRetriever
            max_surface_chunks: Max chunks from surface navigation
            max_suppression_chunks: Max chunks from flow suppression
            max_chars: Max chars per chunk

        Returns:
            Complete dual-mode context as LMIX prose
        """
        from luna9.initiative.dual_mode import DualModeContext

        prose = "# Dual-Mode Retrieval Context\n\n"

        if context.query:
            prose += f"Query: {context.query}\n\n"

        # === COHESIVE FLOW (Surface Navigation) ===
        if context.surface and context.mode_used in ['surface', 'dual']:
            prose += "## Cohesive Flow (Surface Navigation)\n\n"
            prose += "Following the natural semantic flow through connected content:\n\n"

            surface_chunks = context.surface.get('chunks', [])
            for i, chunk in enumerate(surface_chunks[:max_surface_chunks], 1):
                content = chunk.get('content', '')
                if len(content) > max_chars:
                    content = content[:max_chars] + "..."

                score = chunk.get('score', 0.0)
                source = chunk.get('source', 'unknown')
                source_label = "ambient" if source == 'ambient' else "junction"

                prose += f"{i}. [{source_label}] (relevance: {score:.2f}) {content}\n\n"

        # === DISPERSED SIGNALS (Flow Suppression) ===
        if context.suppression and context.mode_used in ['suppression', 'dual']:
            prose += "## Dispersed Signals (Flow Suppression)\n\n"
            prose += self.render_suppression_results(
                context.suppression,
                max_chunks=max_suppression_chunks,
                max_chars=max_chars
            )

        # === INTERPRETATION ===
        if context.mode_used == 'dual':
            prose += "\n## Interpretation\n\n"
            prose += "The cohesive flow shows connected narrative sequences. "
            prose += "The dispersed signals reveal scattered references that appear "
            prose += "throughout but don't form continuous threads. Together, they "
            prose += "provide comprehensive coverage of both connected and scattered content.\n\n"

        return prose
