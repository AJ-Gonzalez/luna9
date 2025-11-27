"""
Possibilities Mapping Component

Maps what's pulling in semantic space:
- High-curvature regions (conceptual junctions)
- Tension threads (unresolved queries)
- Unexplored regions nearby
"""

from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import numpy as np


@dataclass
class Possibilities:
    """
    What's pulling in semantic space.

    Represents attention attractors - regions that might be interesting
    to explore or engage with.
    """
    high_curvature_regions: List[Dict]  # [{position, curvature, description}, ...]
    tension_threads: List[Dict]  # [{query, energy, last_touched}, ...]
    unexplored_nearby: List[Dict]  # [{region, distance}, ...]


class PossibilitiesMapper:
    """
    Maps possibilities from geometric surface.

    Computes high-curvature regions, tension detection, and unexplored areas.
    """

    def __init__(self, domain: 'Domain'):
        """
        Initialize possibilities mapper.

        Args:
            domain: Domain instance to map possibilities from
        """
        self.domain = domain
        self._curvature_cache: Optional[np.ndarray] = None

    def compute_possibilities(
        self,
        current_position: Tuple[float, float],
        radius: float = 0.5,
        top_k: int = 3,
        k_retrieve_per_region: int = 3
    ) -> Possibilities:
        """
        Find what's pulling from current position.

        NOW WITH RETRIEVAL: Each curvature region includes full retrieved content,
        not just geometric metadata.

        Args:
            current_position: Current (u,v) position
            radius: Search radius for nearby regions
            top_k: Number of top regions to return
            k_retrieve_per_region: Messages to retrieve per curvature region (NEW!)

        Returns:
            Possibilities with curvature regions, tension, unexplored areas
        """
        if self.domain.surface is None:
            # No surface = no possibilities
            return Possibilities(
                high_curvature_regions=[],
                tension_threads=[],
                unexplored_nearby=[]
            )

        # Find high-curvature regions (geometry)
        curvature_regions = self._find_high_curvature_regions(
            current_position, radius, top_k
        )

        # For each region, retrieve content (NEW!)
        enriched_regions = []
        for region in curvature_regions:
            if k_retrieve_per_region > 0:
                # Perform retrieval
                retrieval_data = self._describe_and_retrieve_curvature_region(
                    region['position'],
                    k_retrieve=k_retrieve_per_region
                )

                enriched_regions.append({
                    'position': region['position'],
                    'curvature': region['curvature'],
                    'distance': region['distance'],
                    'description': retrieval_data['semantic_label'],
                    'retrieved_content': retrieval_data['retrieved_messages']  # Full text!
                })
            else:
                # Fallback to old behavior (no retrieval)
                enriched_regions.append({
                    **region,
                    'retrieved_content': []  # Empty list maintains structure
                })

        # Detect tension (simplified: queries vs. statements)
        tension_threads = self._detect_tension(top_k)

        # Find unexplored regions (simplified: low-visit areas)
        unexplored = self._find_unexplored(current_position, radius, top_k)

        return Possibilities(
            high_curvature_regions=enriched_regions,
            tension_threads=tension_threads,
            unexplored_nearby=unexplored
        )

    def _find_high_curvature_regions(
        self,
        current_position: Tuple[float, float],
        radius: float,
        top_k: int
    ) -> List[Dict]:
        """
        Find high-curvature regions within radius.

        High curvature indicates conceptual junctions where multiple
        threads converge.

        Args:
            current_position: Current (u,v)
            radius: Search radius
            top_k: Number to return

        Returns:
            List of curvature regions with position, curvature, description
        """
        from luna9.core.surface_math import compute_curvature

        surface = self.domain.surface
        control_points = surface.control_points
        weights = surface.weights

        # Sample grid of points to check curvature
        sample_density = 10  # 10x10 grid
        u_samples = np.linspace(0, 1, sample_density)
        v_samples = np.linspace(0, 1, sample_density)

        curvature_data = []

        for u in u_samples:
            for v in v_samples:
                # Check if within radius of current position
                distance = ((u - current_position[0])**2 +
                           (v - current_position[1])**2)**0.5

                if distance > radius:
                    continue

                # Compute curvature at this point
                try:
                    K, H = compute_curvature(control_points, weights, u, v)
                    # Use mean curvature |H| as measure of conceptual density
                    curvature_magnitude = abs(H)

                    curvature_data.append({
                        'position': (u, v),
                        'curvature': curvature_magnitude,
                        'distance': distance
                    })
                except (ValueError, ZeroDivisionError):
                    # Skip points where curvature computation fails
                    continue

        # Sort by curvature (highest first) and take top_k
        curvature_data.sort(key=lambda x: x['curvature'], reverse=True)

        # Format results
        results = []
        for i, data in enumerate(curvature_data[:top_k]):
            # Try to describe what's at this region
            description = self._describe_curvature_region(data['position'])
            results.append({
                'position': data['position'],
                'curvature': data['curvature'],
                'description': description,
                'distance': data['distance']
            })

        return results

    def _describe_curvature_region(self, position: Tuple[float, float]) -> str:
        """
        Describe what's at a curvature region.

        Simplified: Find nearest message and use its preview.

        Args:
            position: (u,v) position

        Returns:
            Description string
        """
        surface = self.domain.surface

        # Find nearest control point
        min_dist = float('inf')
        nearest_idx = 0

        for i in range(surface.grid_m):
            for j in range(surface.grid_n):
                # Control point grid coordinates
                u_grid = i / (surface.grid_m - 1) if surface.grid_m > 1 else 0.5
                v_grid = j / (surface.grid_n - 1) if surface.grid_n > 1 else 0.5

                dist = ((position[0] - u_grid)**2 + (position[1] - v_grid)**2)**0.5

                if dist < min_dist:
                    min_dist = dist
                    # Map grid (i,j) to message index
                    nearest_idx = i * surface.grid_n + j

        # Get message at that index
        if nearest_idx < len(surface.messages):
            message = surface.messages[nearest_idx]
            preview = message[:60] + "..." if len(message) > 60 else message
            return preview
        else:
            return "unknown region"

    def _describe_and_retrieve_curvature_region(
        self,
        position: Tuple[float, float],
        k_retrieve: int = 3
    ) -> Dict[str, any]:
        """
        Describe AND retrieve content for a curvature region.

        Combines geometry (where we are) with retrieval (what's there).

        Args:
            position: (u,v) position of curvature region
            k_retrieve: Number of messages to retrieve

        Returns:
            Dict with:
            - 'semantic_label': Short description (e.g., "junction near: topic X")
            - 'retrieved_messages': List of dicts with full text + metadata
            - 'query_position': (u, v) used for retrieval
        """
        surface = self.domain.surface

        # 1. Find nearest message (same logic as _describe_curvature_region)
        min_dist = float('inf')
        nearest_idx = 0

        for i in range(surface.grid_m):
            for j in range(surface.grid_n):
                # Control point grid coordinates
                u_grid = i / (surface.grid_m - 1) if surface.grid_m > 1 else 0.5
                v_grid = j / (surface.grid_n - 1) if surface.grid_n > 1 else 0.5

                dist = ((position[0] - u_grid)**2 + (position[1] - v_grid)**2)**0.5

                if dist < min_dist:
                    min_dist = dist
                    # Map grid (i,j) to message index
                    nearest_idx = i * surface.grid_n + j

        # 2. Get message at that index
        if nearest_idx < len(surface.messages):
            message = surface.messages[nearest_idx]
        else:
            message = ""

        # 3. Use message as query to retrieve neighborhood (THIS IS NEW!)
        result = self.domain.query(message, k=k_retrieve, mode='both')

        # 4. Extract full content (use 'sources' for exact matches)
        retrieved_messages = []
        if 'sources' in result and 'messages' in result['sources']:
            messages = result['sources']['messages']
            distances = result['sources'].get('distances', [0.0] * len(messages))
            indices = result['sources'].get('indices', list(range(len(messages))))

            for msg, dist, idx in zip(messages[:k_retrieve],
                                     distances[:k_retrieve],
                                     indices[:k_retrieve]):
                retrieved_messages.append({
                    'text': msg,  # Full text!
                    'distance': dist,
                    'score': 1.0 - dist if dist < 1.0 else 0.0,
                    'metadata': {'index': idx}
                })

        # 5. Create semantic label (short description for overview)
        semantic_label = f"junction near: {message[:40]}..." if message else "unknown region"

        return {
            'semantic_label': semantic_label,
            'retrieved_messages': retrieved_messages,
            'query_position': position
        }

    def _detect_tension(self, top_k: int) -> List[Dict]:
        """
        Detect tension threads (unresolved queries).

        Simplified: Look for messages ending with '?' (questions)
        These represent threads that might want resolution.

        Args:
            top_k: Number of tension threads to return

        Returns:
            List of tension threads with query, energy, last_touched
        """
        surface = self.domain.surface
        tension_threads = []

        for i, message in enumerate(surface.messages):
            # Simple heuristic: messages ending with '?' are queries
            if message.strip().endswith('?'):
                # Energy is arbitrary for now (could be based on how long unanswered)
                # Simplified: all questions have medium energy
                tension_threads.append({
                    'query': message[:80] + "..." if len(message) > 80 else message,
                    'energy': 0.5,  # Medium tension
                    'last_touched': 'unknown',  # Would need timestamp tracking
                    'index': i
                })

        # Return top_k (for now just first k)
        return tension_threads[:top_k]

    def _find_unexplored(
        self,
        current_position: Tuple[float, float],
        radius: float,
        top_k: int
    ) -> List[Dict]:
        """
        Find unexplored regions nearby.

        Simplified: Regions that are geometrically distant from current position
        but still within radius.

        Args:
            current_position: Current (u,v)
            radius: Search radius
            top_k: Number to return

        Returns:
            List of unexplored regions with region name and distance
        """
        # Sample potential unexplored regions
        # Simplified: Check corners and edges of the surface
        candidate_regions = [
            ((0.1, 0.1), "lower-left quadrant"),
            ((0.1, 0.9), "upper-left quadrant"),
            ((0.9, 0.1), "lower-right quadrant"),
            ((0.9, 0.9), "upper-right quadrant"),
            ((0.5, 0.1), "lower edge"),
            ((0.5, 0.9), "upper edge"),
            ((0.1, 0.5), "left edge"),
            ((0.9, 0.5), "right edge"),
        ]

        unexplored = []

        for position, name in candidate_regions:
            distance = ((position[0] - current_position[0])**2 +
                       (position[1] - current_position[1])**2)**0.5

            # Within radius but not too close
            if 0.2 < distance <= radius:
                unexplored.append({
                    'region': name,
                    'distance': distance,
                    'position': position
                })

        # Sort by distance and return top_k
        unexplored.sort(key=lambda x: x['distance'])
        return unexplored[:top_k]

    def invalidate_cache(self) -> None:
        """Invalidate curvature cache when surface changes."""
        self._curvature_cache = None
