"""
Flow Suppression Component

Suppresses cohesive semantic flow to reveal dispersed signals.

The core insight: When narrative content has strong cohesive flow (e.g., Felix's
cottage story in Frankenstein), it dominates the semantic surface. Dispersed
content (e.g., Elizabeth's scattered appearances) gets "washed out" by this flow.

By computing and suppressing the strong flow vectors (normals), we bring down
the dominant signals, allowing previously hidden dispersed content to become
visible - like audio compression/limiting bringing up quiet parts.

Mathematical Foundation:
- Normal vector: n = embedding - S(u,v) (orthogonal component to surface)
- Normal magnitude: ||n|| (strength of deviation from surface)
- Suppression: n' = n - f(||n||) * (n/||n||) where f is suppression function
- Revelation: Content with high ||n'|| after suppression was dispersed
"""

from typing import List, Dict, Optional, Tuple, Callable
import numpy as np
from dataclasses import dataclass

from ..components.domain import Domain


@dataclass
class SuppressionResult:
    """
    Results from flow suppression.

    Contains both the mathematical artifacts (normals, magnitudes) and
    the revealed content chunks for retrieval.
    """
    # Mathematical results
    normals: np.ndarray  # (n, 768) normal vectors before suppression
    magnitudes: np.ndarray  # (n,) normal magnitudes before suppression
    suppressed_normals: np.ndarray  # (n, 768) after suppression
    suppressed_magnitudes: np.ndarray  # (n,) after suppression

    # Revealed content
    revealed_chunks: List[Dict]  # Chunks revealed by suppression

    # Metadata
    suppression_type: str
    parameters: Dict


class FlowSuppressor:
    """
    Suppresses cohesive flow to reveal dispersed signals.

    Uses non-linear suppression of surface normals to bring down
    dominant semantic flow, making scattered content visible.

    Supports two suppression functions:
    - Power law: f(m) = m^β (β > 1)
    - Sigmoid: f(m) = m * sigmoid(γ * m)

    Example:
        >>> domain = Domain.create_from_messages("frankenstein", ...)
        >>> suppressor = FlowSuppressor(domain, suppression_type='power_law', beta=2.5)
        >>> revealed = suppressor.get_revealed_chunks(query="Elizabeth", top_k=10)
    """

    def __init__(
        self,
        domain: Domain,
        suppression_type: str = 'power_law',
        beta: float = 2.5,
        gamma: float = 3.0,
        cache_normals: bool = True
    ):
        """
        Initialize flow suppressor.

        Args:
            domain: Domain with semantic surface to suppress
            suppression_type: 'power_law' or 'sigmoid'
            beta: Power law exponent (β > 1 for stronger suppression)
            gamma: Sigmoid steepness (higher = sharper transition)
            cache_normals: Cache computed normals for reuse
        """
        if domain.surface is None:
            raise ValueError("Domain has no surface - cannot perform flow suppression")

        self.domain = domain
        self.suppression_type = suppression_type
        self.beta = beta
        self.gamma = gamma
        self.cache_normals = cache_normals

        # Cache
        self._cached_normals: Optional[np.ndarray] = None
        self._cached_magnitudes: Optional[np.ndarray] = None

    def compute_normals(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute normal vectors for all control points.

        The "normal" here is the orthogonal component between the actual
        embedding and its projection onto the parametric surface. This
        represents the 766-dimensional space perpendicular to the surface's
        tangent plane at that point.

        High magnitude = strong deviation = cohesive flow
        Low magnitude = weak deviation = dispersed/averaged

        Returns:
            Tuple of:
                - normals: (n, 768) array of orthogonal components
                - magnitudes: (n,) array of normal magnitudes
        """
        # Check cache
        if self.cache_normals and self._cached_normals is not None:
            return self._cached_normals, self._cached_magnitudes

        surface = self.domain.surface
        normals = []
        magnitudes = []

        # Iterate over all control points
        for i in range(surface.grid_m):
            for j in range(surface.grid_n):
                # Get parametric coordinates
                u = i / max(1, surface.grid_m - 1)
                v = j / max(1, surface.grid_n - 1)

                # Get actual embedding at this control point
                msg_idx = surface.provenance['cp_to_msg'][(i, j)]
                embedding = surface.embeddings[msg_idx]

                # Get surface interpolation at (u,v)
                surface_point = surface.evaluate_at(u, v)

                # Normal = orthogonal component = embedding - projection
                normal = embedding - surface_point
                magnitude = np.linalg.norm(normal)

                normals.append(normal)
                magnitudes.append(magnitude)

        normals_array = np.array(normals)
        magnitudes_array = np.array(magnitudes)

        # Cache if enabled
        if self.cache_normals:
            self._cached_normals = normals_array
            self._cached_magnitudes = magnitudes_array

        return normals_array, magnitudes_array

    def suppress_flow(
        self,
        normals: np.ndarray,
        magnitudes: np.ndarray
    ) -> np.ndarray:
        """
        Apply non-linear suppression to normals.

        Suppression functions:
        - Power law: f(m) = m^β where β > 1
          - β=2: quadratic suppression
          - β=2.5: stronger suppression (recommended)
          - β=3: cubic suppression

        - Sigmoid: f(m) = m * sigmoid(γ * m)
          - γ=3.0: moderate transition (recommended)
          - γ=5.0: sharp transition

        The suppression is applied as:
            n' = n - f(||n||) * (n/||n||)

        This reduces the normal by its suppression factor along its direction,
        bringing down strong flows more than weak ones.

        Args:
            normals: (n, 768) normal vectors
            magnitudes: (n,) normal magnitudes

        Returns:
            suppressed_magnitudes: (n,) magnitudes after suppression
        """
        # Compute suppression factors based on type
        if self.suppression_type == 'power_law':
            suppression_factors = magnitudes ** self.beta
        elif self.suppression_type == 'sigmoid':
            sigmoid = 1 / (1 + np.exp(-self.gamma * magnitudes))
            suppression_factors = magnitudes * sigmoid
        else:
            raise ValueError(f"Unknown suppression type: {self.suppression_type}")

        # Apply suppression: n' = n - f(||n||) * (n/||n||)
        suppressed_normals = normals.copy()
        for i in range(len(normals)):
            if magnitudes[i] > 1e-8:  # Avoid division by zero
                direction = normals[i] / magnitudes[i]
                suppressed_normals[i] = normals[i] - suppression_factors[i] * direction

        # Compute new magnitudes
        suppressed_magnitudes = np.linalg.norm(suppressed_normals, axis=1)

        return suppressed_magnitudes

    def get_revealed_chunks(
        self,
        query: Optional[str] = None,
        top_k: int = 10,
        revelation_threshold: Optional[float] = None
    ) -> List[Dict]:
        """
        Get chunks revealed by flow suppression.

        "Revealed" means: chunks that were weak before suppression but
        become relatively strong after suppression. These are the dispersed
        signals that were previously drowned out by cohesive flow.

        Args:
            query: Optional query to focus retrieval (filters by relevance)
            top_k: Number of revealed chunks to return
            revelation_threshold: Percentile threshold for revelation (default: 75)

        Returns:
            List of dicts with:
                - message_idx: chunk index in domain
                - content: chunk text
                - score: revelation strength (suppressed magnitude)
                - position: (u,v) coordinates on surface
                - normal_before: magnitude before suppression
                - normal_after: magnitude after suppression
        """
        if revelation_threshold is None:
            revelation_threshold = 75.0

        surface = self.domain.surface

        # Compute normals (may use cache)
        normals, magnitudes = self.compute_normals()

        # Apply suppression
        suppressed_magnitudes = self.suppress_flow(normals, magnitudes)

        # Find strong flow before suppression
        strong_flow_threshold = np.percentile(magnitudes, revelation_threshold)
        strong_flow_mask = magnitudes > strong_flow_threshold

        # Find strong signals after suppression
        revealed_threshold = np.percentile(suppressed_magnitudes, revelation_threshold)
        revealed_mask = suppressed_magnitudes > revealed_threshold

        # Newly revealed = was weak, now strong
        newly_revealed_mask = revealed_mask & ~strong_flow_mask

        # Build result list
        revealed_chunks = []
        for idx, is_revealed in enumerate(newly_revealed_mask):
            if is_revealed:
                # Map back to control point
                i = idx // surface.grid_n
                j = idx % surface.grid_n

                # Get message index
                msg_idx = surface.provenance['cp_to_msg'][(i, j)]

                # Get message content
                # Access messages through surface.messages (the raw list)
                content = surface.messages[msg_idx]

                # Compute position
                u = i / max(1, surface.grid_m - 1)
                v = j / max(1, surface.grid_n - 1)

                revealed_chunks.append({
                    'message_idx': int(msg_idx),
                    'content': content,
                    'score': float(suppressed_magnitudes[idx]),
                    'position': (float(u), float(v)),
                    'normal_before': float(magnitudes[idx]),
                    'normal_after': float(suppressed_magnitudes[idx])
                })

        # Sort by revelation score (descending)
        revealed_chunks.sort(key=lambda x: x['score'], reverse=True)

        # Optionally filter by query relevance
        if query is not None:
            # Get embedding model from surface
            model = surface._get_embedding_model()
            query_embedding = model.encode([query], show_progress_bar=False)[0]

            # Compute relevance scores
            for chunk in revealed_chunks:
                chunk_embedding = surface.embeddings[chunk['message_idx']]
                similarity = np.dot(query_embedding, chunk_embedding)
                chunk['query_relevance'] = float(similarity)

            # Re-sort by query relevance
            revealed_chunks.sort(key=lambda x: x['query_relevance'], reverse=True)

        # Return top_k
        return revealed_chunks[:top_k]

    def analyze_suppression(self) -> SuppressionResult:
        """
        Perform full suppression analysis.

        Returns complete mathematical artifacts and revealed chunks.
        Useful for debugging and understanding what suppression does.

        Returns:
            SuppressionResult with all normals, magnitudes, and revealed chunks
        """
        # Compute normals
        normals, magnitudes = self.compute_normals()

        # Apply suppression
        suppressed_magnitudes = self.suppress_flow(normals, magnitudes)

        # Reconstruct suppressed normals for completeness
        suppressed_normals = normals.copy()
        if self.suppression_type == 'power_law':
            suppression_factors = magnitudes ** self.beta
        elif self.suppression_type == 'sigmoid':
            sigmoid = 1 / (1 + np.exp(-self.gamma * magnitudes))
            suppression_factors = magnitudes * sigmoid

        for i in range(len(normals)):
            if magnitudes[i] > 1e-8:
                direction = normals[i] / magnitudes[i]
                suppressed_normals[i] = normals[i] - suppression_factors[i] * direction

        # Get revealed chunks
        revealed_chunks = self.get_revealed_chunks(top_k=20)

        return SuppressionResult(
            normals=normals,
            magnitudes=magnitudes,
            suppressed_normals=suppressed_normals,
            suppressed_magnitudes=suppressed_magnitudes,
            revealed_chunks=revealed_chunks,
            suppression_type=self.suppression_type,
            parameters={
                'beta': self.beta,
                'gamma': self.gamma
            }
        )

    def clear_cache(self):
        """Clear cached normals (call after surface rebuild)."""
        self._cached_normals = None
        self._cached_magnitudes = None
