"""
Luna Nine Core - Low-level geometric primitives

This module contains the fundamental building blocks for geometric semantic memory:
- Semantic surfaces (NURBS-based embeddings)
- Surface mathematics (curvature, geodesics, projection)
- Hash indexing (fast O(1) candidate lookup)
"""

from .semantic_surface import SemanticSurface, RetrievalResult
from .hash_index import HashIndex, MessageEntry

__all__ = [
    'SemanticSurface',
    'RetrievalResult',
    'HashIndex',
    'MessageEntry',
]
