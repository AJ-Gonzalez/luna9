"""
State Surfacing Component

Builds current state context from semantic surface:
- Position (u,v coordinates)
- Ambient memories (nearest messages)
- Trajectory (movement pattern)
"""

from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
from collections import deque


@dataclass
class StateContext:
    """
    Current state in semantic space.

    Represents where we are, what's present, and how we've been moving.
    """
    position: Tuple[float, float]  # (u, v) coordinates on surface
    domain_path: str  # Which domain we're in
    ambient_memories: List[Dict]  # Nearest memories with distances
    trajectory: str  # "linear" / "jumping" / "deepening" / "stationary"


class StateSurface:
    """
    Builds state context from domain.

    Tracks position, ambient context, and trajectory across queries.
    """

    def __init__(self, domain: 'Domain'):
        """
        Initialize state surface.

        Args:
            domain: Domain instance to track state for
        """
        self.domain = domain
        self.position_history: deque = deque(maxlen=10)  # Last 10 positions
        self.query_history: deque = deque(maxlen=10)  # Last 10 queries

    def get_current_state(
        self,
        query: str,
        k_ambient: int = 5
    ) -> StateContext:
        """
        Build state context for current query.

        Args:
            query: Query string to project onto surface
            k_ambient: Number of ambient memories to include

        Returns:
            StateContext with position, memories, trajectory
        """
        if self.domain.surface is None:
            raise ValueError("Domain has no surface - cannot compute state")

        # Get embedding for query
        model = self.domain.surface._get_embedding_model()
        embedding = model.encode([query], show_progress_bar=False)[0]

        # Project to surface coordinates
        u, v = self.domain.surface.project_embedding(embedding)
        position = (u, v)

        # Query domain for nearest memories
        result = self.domain.query(query, k=k_ambient, mode='both')

        # Extract ambient memories using interpretation (smooth retrieval)
        # This is more semantically meaningful than exact/sources because it uses
        # Bernstein basis influence weights across the entire surface, not just
        # nearest control points which can fail when query projects to sparse regions
        ambient_memories = []
        if 'interpretation' in result and 'messages' in result['interpretation']:
            messages = result['interpretation']['messages']
            weights = result['interpretation'].get('weights', [1.0] * len(messages))
            indices = result['interpretation'].get('indices', list(range(len(messages))))

            for i, (msg, weight, idx) in enumerate(zip(messages[:k_ambient],
                                                        weights[:k_ambient],
                                                        indices[:k_ambient])):
                ambient_memories.append({
                    'text': msg,
                    'distance': 1.0 - weight,  # Convert weight to distance-like metric
                    'score': weight,  # Influence weight IS the score
                    'metadata': {'index': idx}
                })

        # Detect trajectory
        trajectory = self._detect_trajectory(position)

        # Update history
        self.position_history.append(position)
        self.query_history.append(query)

        return StateContext(
            position=position,
            domain_path=self.domain.path,
            ambient_memories=ambient_memories,
            trajectory=trajectory
        )

    def _detect_trajectory(self, current_position: Tuple[float, float]) -> str:
        """
        Detect movement pattern from position history.

        Simplified trajectory detection:
        - Linear: Small variation in movement direction
        - Jumping: Large jumps between positions
        - Deepening: Moving in a spiral/circular pattern
        - Stationary: Not moving much

        Args:
            current_position: Current (u,v) position

        Returns:
            Trajectory type: "linear", "jumping", "deepening", or "stationary"
        """
        if len(self.position_history) < 2:
            return "stationary"

        # Calculate distances between consecutive positions
        distances = []
        for i in range(len(self.position_history) - 1):
            pos1 = self.position_history[i]
            pos2 = self.position_history[i + 1]
            dist = ((pos2[0] - pos1[0])**2 + (pos2[1] - pos1[1])**2)**0.5
            distances.append(dist)

        # Add distance to current position
        last_pos = self.position_history[-1]
        current_dist = ((current_position[0] - last_pos[0])**2 +
                       (current_position[1] - last_pos[1])**2)**0.5
        distances.append(current_dist)

        # Compute average distance
        avg_distance = sum(distances) / len(distances)
        max_distance = max(distances)

        # Simple heuristics for trajectory classification
        if avg_distance < 0.1:
            # Not moving much
            return "stationary"
        elif max_distance > 0.5:
            # Large jumps
            return "jumping"
        elif avg_distance < 0.2:
            # Small, consistent movements - could be deepening into a region
            # Check if we're circling back (simplified: look at distance to earliest position)
            if len(self.position_history) >= 3:
                first_pos = self.position_history[0]
                dist_to_first = ((current_position[0] - first_pos[0])**2 +
                                (current_position[1] - first_pos[1])**2)**0.5
                if dist_to_first < avg_distance * 2:
                    return "deepening"
            return "linear"
        else:
            # Moderate, linear progression
            return "linear"

    def update_trajectory(self, query: str, position: Tuple[float, float]) -> None:
        """
        Update trajectory tracking with new query/position.

        Args:
            query: Query text
            position: (u,v) position
        """
        self.position_history.append(position)
        self.query_history.append(query)

    def get_position_history(self) -> List[Tuple[float, float]]:
        """Get position history."""
        return list(self.position_history)

    def get_query_history(self) -> List[str]:
        """Get query history."""
        return list(self.query_history)

    def clear_history(self) -> None:
        """Clear position and query history."""
        self.position_history.clear()
        self.query_history.clear()
