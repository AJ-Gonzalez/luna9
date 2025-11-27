"""
Dual-Mode Retrieval Coordinator

Combines two complementary retrieval approaches:
1. Surface Navigation: Follows cohesive semantic flow (State + Possibilities)
2. Flow Suppression: Reveals dispersed signals hidden by dominant flow

The insight: Different content has different structure. Cohesive narratives
benefit from surface navigation. Dispersed content (scattered character
appearances, themes) needs flow suppression to become visible.

By using both modes together, we get comprehensive retrieval that handles
both cohesive and dispersed content effectively.
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np

from ..components.domain import Domain
from .state import StateSurface
from .possibilities import PossibilitiesMapper
from .flow_suppressor import FlowSuppressor


@dataclass
class DualModeContext:
    """
    Context from dual-mode retrieval.

    Contains results from both surface navigation and flow suppression,
    plus a combined view for comprehensive understanding.
    """
    # Surface navigation results
    surface: Optional[Dict] = None

    # Flow suppression results
    suppression: Optional[List[Dict]] = None

    # Combined/fused results
    combined: Optional[List[Dict]] = None

    # Metadata
    mode_used: str = 'dual'  # 'surface', 'suppression', or 'dual'
    surface_weight: float = 0.5
    suppression_weight: float = 0.5

    # Query info
    query: Optional[str] = None


class DualModeRetriever:
    """
    Coordinates dual-mode retrieval: surface navigation + flow suppression.

    Combines cohesive flow (surface navigation) with dispersed signals
    (flow suppression) for comprehensive retrieval.

    The two modes are complementary:
    - Surface navigation: Good for cohesive narratives, storylines
    - Flow suppression: Good for scattered characters, themes, entities

    Example:
        >>> domain = Domain.create_from_messages("book", ...)
        >>> retriever = DualModeRetriever(domain)
        >>> context = retriever.retrieve("Who is Elizabeth?", mode='dual')
        >>> # context.surface has cohesive flow results
        >>> # context.suppression has dispersed signal results
        >>> # context.combined has both fused together
    """

    def __init__(
        self,
        domain: Domain,
        surface_weight: float = 0.5,
        suppression_weight: float = 0.5,
        suppression_type: str = 'power_law',
        use_sigmoid: bool = False,
        beta: float = 2.5,
        gamma: float = 3.0,
        auto_mode: bool = False
    ):
        """
        Initialize dual-mode retriever.

        Args:
            domain: Domain with semantic surface
            surface_weight: Weight for surface navigation results (α)
            suppression_weight: Weight for flow suppression results (β)
            suppression_type: Base suppression type ('power_law' or 'sigmoid')
            use_sigmoid: If True, override suppression_type to use sigmoid
            beta: Power law exponent (used if power_law)
            gamma: Sigmoid steepness (used if sigmoid)
            auto_mode: Auto-detect best mode from query (not implemented yet)
        """
        if domain.surface is None:
            raise ValueError("Domain has no surface - cannot perform dual-mode retrieval")

        self.domain = domain
        self.surface_weight = surface_weight
        self.suppression_weight = suppression_weight
        self.auto_mode = auto_mode

        # Initialize components
        self.state_surface = StateSurface(domain)
        self.possibilities_mapper = PossibilitiesMapper(domain)

        # Override suppression_type if use_sigmoid is True
        if use_sigmoid:
            suppression_type = 'sigmoid'

        self.flow_suppressor = FlowSuppressor(
            domain=domain,
            suppression_type=suppression_type,
            beta=beta,
            gamma=gamma,
            cache_normals=True
        )

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        mode: str = 'dual'
    ) -> DualModeContext:
        """
        Retrieve using dual-mode approach.

        Args:
            query: Search query
            top_k: Results to return per mode
            mode: Which mode(s) to use:
                - 'surface': Only surface navigation
                - 'suppression': Only flow suppression
                - 'dual': Both modes (recommended)

        Returns:
            DualModeContext with results from requested mode(s)
        """
        if mode == 'surface':
            # Only surface navigation
            surface_results = self._get_surface_results(query, top_k)
            return DualModeContext(
                surface=surface_results,
                suppression=None,
                combined=surface_results.get('chunks', []),
                mode_used='surface',
                surface_weight=1.0,
                suppression_weight=0.0,
                query=query
            )

        elif mode == 'suppression':
            # Only flow suppression
            suppression_results = self._get_suppression_results(query, top_k)
            return DualModeContext(
                surface=None,
                suppression=suppression_results,
                combined=suppression_results,
                mode_used='suppression',
                surface_weight=0.0,
                suppression_weight=1.0,
                query=query
            )

        else:  # 'dual'
            # Get both
            surface_results = self._get_surface_results(query, top_k)
            suppression_results = self._get_suppression_results(query, top_k)

            # Combine with weights
            combined = self._combine_results(
                surface_results,
                suppression_results,
                self.surface_weight,
                self.suppression_weight
            )

            return DualModeContext(
                surface=surface_results,
                suppression=suppression_results,
                combined=combined,
                mode_used='dual',
                surface_weight=self.surface_weight,
                suppression_weight=self.suppression_weight,
                query=query
            )

    def _get_surface_results(self, query: str, top_k: int) -> Dict:
        """
        Get surface navigation results.

        Uses State + Possibilities to follow cohesive semantic flow.

        Returns:
            Dict with:
                - state: StateContext (position, ambient memories, trajectory)
                - possibilities: List of possibility regions
                - chunks: Unified list of retrieved chunks with scores
        """
        # Get current state
        state = self.state_surface.get_current_state(query, k_ambient=top_k)

        # Get possibilities (curvature-based junctions)
        possibilities = self.possibilities_mapper.compute_possibilities(
            current_position=state.position,
            top_k=top_k
        )

        # Collect chunks with scores
        chunks = []

        # Add ambient memories from state
        for mem in state.ambient_memories:
            chunks.append({
                'content': mem['text'],
                'score': mem['score'],
                'source': 'ambient',
                'metadata': mem.get('metadata', {})
            })

        # Add possibility regions (from high curvature areas)
        for region in possibilities.high_curvature_regions:
            # Extract content from retrieved_content if available
            retrieved = region.get('retrieved_content', [])
            for item in retrieved:
                chunks.append({
                    'content': item.get('text', ''),
                    'score': 1.0 - item.get('distance', 0.5),  # Convert distance to score
                    'source': 'possibility',
                    'metadata': {
                        'position': region.get('position'),
                        'curvature': region.get('curvature')
                    }
                })

        # Deduplicate by content, preserving order (ambient first, then junctions)
        seen_content = set()
        deduped_chunks = []
        for chunk in chunks:
            content_key = chunk['content'][:100]  # Use first 100 chars as key
            if content_key not in seen_content:
                seen_content.add(content_key)
                deduped_chunks.append(chunk)

        # DON'T sort by score - ambient memories are query-specific and should stay first
        # Junction chunks are position-specific (curvature regions) not query-specific
        # Keeping ambient first ensures query-relevant content appears before static regions

        return {
            'state': state,
            'possibilities': possibilities,
            'chunks': deduped_chunks[:top_k]
        }

    def _get_suppression_results(self, query: str, top_k: int) -> List[Dict]:
        """
        Get flow suppression results.

        Uses FlowSuppressor to reveal dispersed signals.

        Returns:
            List of revealed chunks with revelation scores
        """
        revealed = self.flow_suppressor.get_revealed_chunks(
            query=query,
            top_k=top_k
        )

        # Add source metadata
        for chunk in revealed:
            chunk['source'] = 'suppression'

        return revealed

    def _combine_results(
        self,
        surface_results: Dict,
        suppression_results: List[Dict],
        alpha: float,
        beta: float
    ) -> List[Dict]:
        """
        Combine surface + suppression results.

        Strategy: Distinct sections (Option C from architecture doc)
        - Surface results first (cohesive flow)
        - Then suppression results (dispersed signals)
        - This is clearest for LLMs to understand

        Future: Could implement weighted score fusion or interleaving.

        Args:
            surface_results: Dict from _get_surface_results
            suppression_results: List from _get_suppression_results
            alpha: Surface weight (for future weighted fusion)
            beta: Suppression weight (for future weighted fusion)

        Returns:
            Combined list of chunks with source labels
        """
        combined = []

        # Add surface chunks (cohesive flow)
        surface_chunks = surface_results.get('chunks', [])
        for chunk in surface_chunks:
            combined.append({
                **chunk,
                'mode': 'surface'
            })

        # Add suppression chunks (dispersed signals)
        # Avoid duplicates by checking content overlap
        seen_content = {c['content'][:100] for c in combined}
        for chunk in suppression_results:
            content_key = chunk['content'][:100]
            if content_key not in seen_content:
                combined.append({
                    **chunk,
                    'mode': 'suppression'
                })
                seen_content.add(content_key)

        return combined


def create_dual_mode_retriever(
    domain: Domain,
    use_sigmoid: bool = False,
    **kwargs
) -> DualModeRetriever:
    """
    Convenience factory for creating DualModeRetriever.

    Args:
        domain: Domain with semantic surface
        use_sigmoid: Use sigmoid suppression instead of power law
        **kwargs: Additional parameters for DualModeRetriever

    Returns:
        Configured DualModeRetriever
    """
    return DualModeRetriever(
        domain=domain,
        use_sigmoid=use_sigmoid,
        **kwargs
    )
