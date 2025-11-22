"""
Semantic Surface: Dual-mode retrieval system for conversation memory.

Combines smooth surface navigation (interpretation/knowledge) with exact
control point retrieval (reference/provenance).

"Thinking with a book in hand" - navigate semantic space while maintaining
exact source references.
"""

import time
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass

import numpy as np
from sentence_transformers import SentenceTransformer

from .surface_math import (
    evaluate_surface,
    project_to_surface,
    geodesic_distance,
    compute_curvature,
    bernstein_basis,
    semantic_displacement,
    compute_path_curvature
)
from ..utils.performance import log_performance

# Try to import JIT-compiled fast versions
try:
    from .surface_math_fast import (
        project_to_surface_fast,
        compute_influence_fast,
        find_nearest_control_points_fast
    )
    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False
    project_to_surface_fast = project_to_surface  # Fallback to slow version
    compute_influence_fast = None
    find_nearest_control_points_fast = None


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
                       - 16 messages → 4x4
                       - 12 messages → 3x4
                       - 9 messages → 3x3
                       - etc.
        """
        self.messages = messages
        self.model_name = model_name
        self._embedding_model = None  # Lazy-loaded
        self._dirty = False  # Track if surface needs rebuild
        self._pending_messages = []  # Buffer for new messages
        num_messages = len(messages)

        # Infer grid shape if not provided
        if grid_shape is None:
            grid_shape = self._infer_grid_shape(num_messages)

        self.grid_m, self.grid_n = grid_shape
        expected_count = self.grid_m * self.grid_n

        assert num_messages == expected_count, \
            f"Expected {expected_count} messages for {self.grid_m}x{self.grid_n} grid, got {num_messages}"

        # Embed messages if not provided
        if embeddings is None:
            print(f"Embedding {len(messages)} messages with {model_name}...")
            start = time.perf_counter()
            model = SentenceTransformer(model_name)
            embeddings = model.encode(messages, show_progress_bar=False)
            duration = (time.perf_counter() - start) * 1000
            log_performance("embed_initial", duration,
                          message_count=len(messages),
                          model=model_name,
                          embedding_dim=embeddings.shape[1])
            print(f"  Created embeddings: {embeddings.shape}")

        self.embeddings = embeddings
        self.embedding_dim = embeddings.shape[1]

        # Arrange as mxn control points
        self.control_points = embeddings.reshape(self.grid_m, self.grid_n, self.embedding_dim)
        self.weights = np.ones((self.grid_m, self.grid_n))

        # Build provenance mappings
        self.provenance = self._build_provenance()

        print("Created semantic surface:")
        print(f"  Control points: {self.grid_m}x{self.grid_n}")
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

        # Fallback: 1xn (degenerate but works)
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
        # Use JIT-compiled version if available
        if _HAS_NUMBA and compute_influence_fast is not None:
            # Fast path: JIT-compiled influence computation
            influence_grid = compute_influence_fast(self.weights, u, v)

            # Convert to list with message indices
            influence_weights = []
            for i in range(self.grid_m):
                for j in range(self.grid_n):
                    msg_idx = self.provenance['cp_to_msg'][(i, j)]
                    influence_weights.append((msg_idx, influence_grid[i, j]))
        else:
            # Fallback: Python loop
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
        k: int = 4,
        hash_index=None
    ) -> List[Tuple[int, int, int, float]]:
        """
        Find k nearest control points in parameter space.

        Control point [i, j] is located at (i/(m-1), j/(n-1)) in UV space.
        Computes Euclidean distance in parameter space.

        Args:
            u, v: Query position in parameter space
            k: Number of nearest neighbors to return
            hash_index: Optional HashIndex for O(1) candidate retrieval.
                       If provided, uses hash bucketing for ~10-40x speedup.
                       If None, falls back to O(m*n) full scan.

        Returns:
            List of (i, j, msg_idx, distance) sorted by distance
        """
        # Fast path: Use hash index if available
        if hash_index is not None:
            # Adaptive search radius based on grid density
            # For sparse grids (< 16x16), need larger radius to find control points
            # CP spacing in UV: ~1/(grid_size-1)
            # Quantization spacing: 1/255
            # Need radius to span at least one CP spacing
            grid_size = max(self.grid_m, self.grid_n)
            if grid_size < 16:
                # Sparse grid: need large radius
                search_radius = 20  # ~80/255 ≈ 0.31 in UV space
            elif grid_size < 32:
                # Medium grid: moderate radius
                search_radius = 10  # ~40/255 ≈ 0.16 in UV space
            else:
                # Dense grid: small radius is fine
                search_radius = 3   # ~12/255 ≈ 0.05 in UV space

            entries = hash_index.query(u, v, k=k, search_radius=search_radius)

            # Convert MessageEntry objects to (i, j, msg_idx, distance) format
            result = []
            for entry in entries:
                i, j = self.provenance['msg_to_cp'][entry.message_id]
                result.append((i, j, entry.message_id, entry.distance_to(u, v)))

            return result

        # Fallback: Full scan
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
        max_projection_iterations: int = 50,
        hash_index=None
    ) -> RetrievalResult:
        """
        Query the semantic surface with dual-mode retrieval.

        Args:
            query_text: Text query to search for
            k: Number of results to return
            max_projection_iterations: Max iterations for projection
            hash_index: Optional HashIndex for O(1) candidate retrieval.
                       Enables ~10-40x speedup for nearest neighbor search.
                       Also provides warm start for projection (3-5x speedup).

        Returns:
            RetrievalResult with both smooth and exact retrieval
        """
        start = time.perf_counter()

        # Ensure surface is up-to-date
        self.ensure_built()

        # Embed query
        embed_start = time.perf_counter()
        model = self._get_embedding_model()
        query_embedding = model.encode([query_text], show_progress_bar=False)[0]
        embed_duration = (time.perf_counter() - embed_start) * 1000

        # Get initial guess from hash index if available
        if hash_index is not None:
            u_init, v_init = hash_index.get_initial_guess(self, query_embedding)
        else:
            # Default: start from center
            u_init, v_init = 0.5, 0.5

        # Project to surface (use JIT-compiled version if available)
        project_start = time.perf_counter()
        u, v, iterations = project_to_surface_fast(
            query_embedding,
            self.control_points,
            self.weights,
            u_init=u_init,
            v_init=v_init,
            max_iterations=max_projection_iterations
        )
        project_duration = (time.perf_counter() - project_start) * 1000

        # Compute influence (smooth retrieval)
        influence = self.compute_influence(u, v)

        # Find nearest control points (exact retrieval) - uses hash index if available
        nearest = self.nearest_control_points(u, v, k=k, hash_index=hash_index)

        # Compute curvature at query point
        K, H = compute_curvature(self.control_points, self.weights, u, v)

        total_duration = (time.perf_counter() - start) * 1000

        log_performance("query", total_duration,
                      grid_size=f"{self.grid_m}x{self.grid_n}",
                      message_count=len(self.messages),
                      k=k,
                      projection_iterations=iterations,
                      embed_ms=f"{embed_duration:.2f}",
                      project_ms=f"{project_duration:.2f}")

        return RetrievalResult(
            uv=(u, v),
            influence=influence,
            nearest_control_points=nearest,
            curvature=(K, H)
        )

    def evaluate_at(self, u: float, v: float) -> np.ndarray:
        """Evaluate surface at given UV coordinates."""
        return evaluate_surface(self.control_points, self.weights, u, v)

    def project_embedding(
        self,
        embedding: np.ndarray,
        max_iterations: int = 50
    ) -> Tuple[float, float]:
        """
        Project an embedding onto the semantic surface to get (u,v) coordinates.

        This is useful for hash bucketing and geometric operations that need
        surface coordinates without full retrieval.

        Args:
            embedding: Embedding vector to project (shape: (embedding_dim,))
            max_iterations: Maximum projection iterations

        Returns:
            (u, v): Surface coordinates in [0,1] x [0,1]

        Example:
            >>> surface = SemanticSurface(messages)
            >>> model = SentenceTransformer('all-MiniLM-L6-v2')
            >>> query_emb = model.encode(["new message"])[0]
            >>> u, v = surface.project_embedding(query_emb)
            >>> # Use (u,v) for hash bucketing or geometric operations
        """
        # Ensure surface is up-to-date
        self.ensure_built()

        u, v, _ = project_to_surface_fast(
            embedding,
            self.control_points,
            self.weights,
            max_iterations=max_iterations
        )

        return (u, v)

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


    def _get_embedding_model(self):
        """Lazy-load embedding model."""
        if self._embedding_model is None:
            self._embedding_model = SentenceTransformer(self.model_name)
        return self._embedding_model

    def append_message(
        self,
        message: str,
        metadata: Optional[Dict] = None,
        rebuild_threshold: float = 0.1
    ) -> None:
        """
        Append a single message to the surface.

        Messages are buffered and surface is rebuilt when buffer exceeds
        rebuild_threshold (as fraction of current size) or on next query.

        Args:
            message: Text message to add
            metadata: Optional metadata dict for this message
            rebuild_threshold: Rebuild when pending >= threshold * current_size
        """
        self._pending_messages.append({'text': message, 'metadata': metadata})
        self._dirty = True

        # Auto-rebuild if buffer exceeds threshold
        current_size = len(self.messages)
        pending_size = len(self._pending_messages)

        if pending_size >= max(1, int(current_size * rebuild_threshold)):
            self._rebuild_surface()

    def append_messages(
        self,
        messages: List[str],
        metadata: Optional[List[Dict]] = None,
        rebuild_threshold: float = 0.1
    ) -> None:
        """
        Append multiple messages to the surface (batch operation).

        Args:
            messages: List of text messages to add
            metadata: Optional list of metadata dicts (same length as messages)
            rebuild_threshold: Rebuild when pending >= threshold * current_size
        """
        if metadata is None:
            metadata = [None] * len(messages)

        for msg, meta in zip(messages, metadata):
            self._pending_messages.append({'text': msg, 'metadata': meta})

        self._dirty = True

        # Auto-rebuild if buffer exceeds threshold
        current_size = len(self.messages)
        pending_size = len(self._pending_messages)

        if pending_size >= max(1, int(current_size * rebuild_threshold)):
            self._rebuild_surface()

    def _rebuild_surface(self) -> None:
        """
        Rebuild surface with pending messages using lazy incremental update.

        This method implements the core incremental update strategy for semantic
        surfaces. Instead of rebuilding on every message add, we buffer pending
        messages and rebuild only when the buffer crosses a threshold.

        Algorithm:
        1. Merge pending messages into main message list
        2. Re-embed ALL messages (optimization opportunity: only embed new ones)
        3. Infer new grid dimensions to accommodate additional messages
        4. Reshape embeddings into new control point grid
        5. Rebuild provenance mappings for new grid structure
        6. Clear pending buffer and mark surface as clean

        Performance tracking:
        - Logs embedding time separately (dominant cost)
        - Logs total rebuild time with before/after metrics
        - Reports grid dimension changes

        Side effects:
        - Updates self.messages, self.embeddings, self.control_points
        - Updates self.grid_m, self.grid_n, self.provenance
        - Clears self._pending_messages and sets self._dirty = False

        Note: Current implementation re-embeds all messages for simplicity.
        Future optimization could cache old embeddings and only embed new ones.
        """
        if not self._pending_messages:
            return

        start = time.perf_counter()
        old_size = len(self.messages)
        old_grid = (self.grid_m, self.grid_n)
        pending_count = len(self._pending_messages)

        print(f"Rebuilding surface: {old_size} -> {old_size + pending_count} messages")

        # Add pending messages to messages list
        for pending in self._pending_messages:
            self.messages.append(pending['text'])

        # Recompute embeddings for ALL messages
        # (Could optimize to only embed new ones, but simpler for now)
        embed_start = time.perf_counter()
        model = self._get_embedding_model()
        self.embeddings = model.encode(self.messages, show_progress_bar=False)
        embed_duration = (time.perf_counter() - embed_start) * 1000

        log_performance("embed_rebuild", embed_duration,
                      message_count=len(self.messages),
                      new_messages=pending_count)

        # Infer new grid shape
        num_messages = len(self.messages)
        self.grid_m, self.grid_n = self._infer_grid_shape(num_messages)

        # Reshape control points
        self.control_points = self.embeddings.reshape(
            self.grid_m, self.grid_n, self.embedding_dim
        )
        self.weights = np.ones((self.grid_m, self.grid_n))

        # Rebuild provenance
        self.provenance = self._build_provenance()

        # Clear pending buffer
        self._pending_messages = []
        self._dirty = False

        total_duration = (time.perf_counter() - start) * 1000
        log_performance("rebuild_surface", total_duration,
                      old_size=old_size,
                      new_size=len(self.messages),
                      old_grid=f"{old_grid[0]}x{old_grid[1]}",
                      new_grid=f"{self.grid_m}x{self.grid_n}",
                      pending_count=pending_count)

        print(f"  New grid: {self.grid_m}x{self.grid_n}")

    def ensure_built(self) -> None:
        """
        Ensure surface is up-to-date (rebuild if dirty).

        Call before querying to ensure pending messages are incorporated.
        """
        if self._dirty:
            self._rebuild_surface()


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
