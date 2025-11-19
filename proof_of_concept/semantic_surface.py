"""
Semantic Surface: Dual-mode retrieval system for conversation memory.

Combines smooth surface navigation (interpretation/knowledge) with exact
control point retrieval (reference/provenance).

"Thinking with a book in hand" - navigate semantic space while maintaining
exact source references.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer

from surface_math import (
    evaluate_surface,
    project_to_surface,
    geodesic_distance,
    compute_curvature,
    bernstein_basis,
    semantic_displacement,
    compute_path_curvature
)


@dataclass
class RetrievalResult:
    """Result from querying the semantic surface."""
    uv: Tuple[float, float]  # Query position on surface

    # Smooth retrieval (interpretation)
    influence: List[Tuple[int, float]]  # [(msg_idx, weight), ...] sorted by weight

    # Exact retrieval (reference)
    nearest_control_points: List[Tuple[int, int, int, float]]  # [(i, j, msg_idx, distance), ...]

    # Geometric context
    curvature: Tuple[float, float]  # (Gaussian, Mean) curvature at query point

    def get_messages(self, messages: List[str], mode: str = 'both', k: int = 5) -> Dict:
        """
        Retrieve messages based on mode.

        Args:
            messages: Original message list
            mode: 'smooth', 'exact', or 'both'
            k: Number of messages to return

        Returns:
            Dict with retrieved messages and metadata
        """
        result = {'uv': self.uv, 'curvature': self.curvature}

        if mode in ['smooth', 'both']:
            result['interpretation'] = {
                'messages': [messages[idx] for idx, _ in self.influence[:k]],
                'weights': [weight for _, weight in self.influence[:k]],
                'indices': [idx for idx, _ in self.influence[:k]]
            }

        if mode in ['exact', 'both']:
            result['sources'] = {
                'messages': [messages[msg_idx] for _, _, msg_idx, _ in self.nearest_control_points[:k]],
                'distances': [dist for _, _, _, dist in self.nearest_control_points[:k]],
                'indices': [msg_idx for _, _, msg_idx, _ in self.nearest_control_points[:k]]
            }

        return result


class SemanticSurface:
    """
    Semantic surface for conversation memory with dual-mode retrieval.

    Combines:
    - Smooth surface navigation (Bézier surface, interpretation)
    - Exact control point retrieval (discrete provenance, sources)

    The surface represents the semantic space BETWEEN messages, while
    control points anchor exact message locations.
    """

    def __init__(
        self,
        messages: List[str],
        embeddings: Optional[np.ndarray] = None,
        model_name: str = 'all-MiniLM-L6-v2',
        grid_shape: Optional[Tuple[int, int]] = None
    ):
        """
        Create semantic surface from messages.

        Args:
            messages: List of text messages
            embeddings: Pre-computed embeddings (n, d) or None to compute
            model_name: sentence-transformers model to use if computing embeddings
            grid_shape: Optional (m, n) tuple for surface dimensions. If None, infers:
                       - 16 messages → 4×4
                       - 12 messages → 3×4
                       - 9 messages → 3×3
                       - etc.
        """
        self.messages = messages
        self.model_name = model_name
        num_messages = len(messages)

        # Infer grid shape if not provided
        if grid_shape is None:
            grid_shape = self._infer_grid_shape(num_messages)

        self.grid_m, self.grid_n = grid_shape
        expected_count = self.grid_m * self.grid_n

        assert num_messages == expected_count, \
            f"Expected {expected_count} messages for {self.grid_m}×{self.grid_n} grid, got {num_messages}"

        # Embed messages if not provided
        if embeddings is None:
            print(f"Embedding {len(messages)} messages with {model_name}...")
            model = SentenceTransformer(model_name)
            embeddings = model.encode(messages, show_progress_bar=False)
            print(f"  Created embeddings: {embeddings.shape}")

        self.embeddings = embeddings
        self.embedding_dim = embeddings.shape[1]

        # Arrange as m×n control points
        self.control_points = embeddings.reshape(self.grid_m, self.grid_n, self.embedding_dim)
        self.weights = np.ones((self.grid_m, self.grid_n))

        # Build provenance mappings
        self.provenance = self._build_provenance()

        print(f"Created semantic surface:")
        print(f"  Control points: {self.grid_m}×{self.grid_n}")
        print(f"  Embedding dim: {self.embedding_dim}")
        print(f"  Messages: {len(messages)}")

    def _infer_grid_shape(self, num_messages: int) -> Tuple[int, int]:
        """
        Infer reasonable grid shape for given number of messages.

        Tries to create as square-ish a grid as possible.
        """
        import math

        # Common cases
        if num_messages == 16:
            return (4, 4)
        elif num_messages == 12:
            return (3, 4)
        elif num_messages == 9:
            return (3, 3)
        elif num_messages == 20:
            return (4, 5)
        elif num_messages == 25:
            return (5, 5)

        # General case: find factors closest to square
        sqrt_n = int(math.sqrt(num_messages))

        for m in range(sqrt_n, 0, -1):
            if num_messages % m == 0:
                n = num_messages // m
                return (m, n)

        # Fallback: 1×n (degenerate but works)
        return (1, num_messages)

    def _build_provenance(self) -> Dict:
        """Build bidirectional provenance mappings."""
        control_point_to_message = {}
        message_to_control_point = {}

        for idx, (i, j) in enumerate([
            (i, j) for i in range(self.grid_m) for j in range(self.grid_n)
        ]):
            control_point_to_message[(i, j)] = idx
            message_to_control_point[idx] = (i, j)

        return {
            'cp_to_msg': control_point_to_message,
            'msg_to_cp': message_to_control_point
        }

    def compute_influence(self, u: float, v: float) -> List[Tuple[int, float]]:
        """
        Compute influence of each control point at surface position (u, v).

        Uses Bernstein basis functions to determine how much each control
        point contributes to the surface at this location.

        Args:
            u, v: Surface coordinates

        Returns:
            List of (message_index, influence_weight) sorted by weight descending
        """
        # Compute Bernstein basis values for all control points
        influence_weights = []

        degree_u = self.grid_m - 1
        degree_v = self.grid_n - 1

        for i in range(self.grid_m):
            for j in range(self.grid_n):
                # Bernstein basis for this control point
                basis_u = bernstein_basis(i, degree_u, u)
                basis_v = bernstein_basis(j, degree_v, v)

                # Combined influence (product of basis functions times weight)
                influence = self.weights[i, j] * basis_u * basis_v

                # Get message index for this control point
                msg_idx = self.provenance['cp_to_msg'][(i, j)]

                influence_weights.append((msg_idx, influence))

        # Normalize so weights sum to 1
        total_influence = sum(weight for _, weight in influence_weights)
        influence_weights = [
            (idx, weight / total_influence)
            for idx, weight in influence_weights
        ]

        # Sort by weight descending
        influence_weights.sort(key=lambda x: x[1], reverse=True)

        return influence_weights

    def nearest_control_points(
        self,
        u: float,
        v: float,
        k: int = 4
    ) -> List[Tuple[int, int, int, float]]:
        """
        Find k nearest control points in parameter space.

        Control point [i, j] is located at (i/(m-1), j/(n-1)) in UV space.
        Computes Euclidean distance in parameter space.

        Args:
            u, v: Query position in parameter space
            k: Number of nearest neighbors to return

        Returns:
            List of (i, j, msg_idx, distance) sorted by distance
        """
        distances = []

        for i in range(self.grid_m):
            for j in range(self.grid_n):
                # Control point position in parameter space
                cp_u = i / max(1, self.grid_m - 1)
                cp_v = j / max(1, self.grid_n - 1)

                # Euclidean distance in parameter space
                dist = np.sqrt((u - cp_u)**2 + (v - cp_v)**2)

                # Get message index
                msg_idx = self.provenance['cp_to_msg'][(i, j)]

                distances.append((i, j, msg_idx, dist))

        # Sort by distance
        distances.sort(key=lambda x: x[3])

        return distances[:k]

    def query(
        self,
        query_text: str,
        k: int = 5,
        max_projection_iterations: int = 50
    ) -> RetrievalResult:
        """
        Query the semantic surface with dual-mode retrieval.

        Args:
            query_text: Text query to search for
            k: Number of results to return
            max_projection_iterations: Max iterations for projection

        Returns:
            RetrievalResult with both smooth and exact retrieval
        """
        # Embed query
        model = SentenceTransformer(self.model_name)
        query_embedding = model.encode([query_text], show_progress_bar=False)[0]

        # Project to surface
        u, v, iterations = project_to_surface(
            query_embedding,
            self.control_points,
            self.weights,
            max_iterations=max_projection_iterations
        )

        # Compute influence (smooth retrieval)
        influence = self.compute_influence(u, v)

        # Find nearest control points (exact retrieval)
        nearest = self.nearest_control_points(u, v, k=k)

        # Compute curvature at query point
        K, H = compute_curvature(self.control_points, self.weights, u, v)

        return RetrievalResult(
            uv=(u, v),
            influence=influence,
            nearest_control_points=nearest,
            curvature=(K, H)
        )

    def evaluate_at(self, u: float, v: float) -> np.ndarray:
        """Evaluate surface at given UV coordinates."""
        return evaluate_surface(self.control_points, self.weights, u, v)

    def compute_geodesic(
        self,
        uv1: Tuple[float, float],
        uv2: Tuple[float, float],
        num_steps: int = 100
    ) -> float:
        """Compute geodesic distance between two points on surface."""
        return geodesic_distance(
            self.control_points,
            self.weights,
            uv1,
            uv2,
            num_steps=num_steps
        )

    def compute_displacement(
        self,
        uv1: Tuple[float, float],
        uv2: Tuple[float, float]
    ) -> dict:
        """
        Compute net semantic displacement vector between two points.

        Returns directional information (not just scalar distance) showing
        in what semantic dimensions the points differ.

        Returns:
            dict with 'vector', 'distance', 'direction'
        """
        return semantic_displacement(
            self.control_points,
            self.weights,
            uv1,
            uv2
        )

    def compute_path_curvature(
        self,
        uv1: Tuple[float, float],
        uv2: Tuple[float, float],
        num_steps: int = 100
    ) -> dict:
        """
        Compute path curvature metrics showing semantic transition complexity.

        Measures how much the path twists/turns between two points.
        High curvature suggests relationship requires intermediate concepts.

        Returns:
            dict with 'arc_length', 'total_curvature', 'max_curvature',
            'mean_curvature', 'curvature_profile', 'path_points'
        """
        return compute_path_curvature(
            self.control_points,
            self.weights,
            uv1,
            uv2,
            num_steps=num_steps
        )


def create_surface_from_conversation(
    messages: List[str],
    model_name: str = 'all-MiniLM-L6-v2'
) -> SemanticSurface:
    """
    Convenience function to create semantic surface from conversation.

    Args:
        messages: List of 16 messages
        model_name: Embedding model to use

    Returns:
        SemanticSurface ready for querying
    """
    return SemanticSurface(messages, model_name=model_name)
