"""
Luna Nine - Geometric Memory for Language Models

Navigate semantic space through parametric surfaces.

Universal semantic matching engine for:
- Memory and context management
- Product positioning analysis
- Job/candidate matching
- Feature comparison
- Competitive analysis
- Market segmentation
- Any semantic relationship inference

Key Features:
- Sub-linear query scaling (O(√n to n^0.7))
- Perfect linear storage (1.54 KB per message)
- Geometric inference (no hallucination)
- Interpretable scores (0.0-1.0 scales)
- Auto-scaling method selection
"""

__version__ = "0.1.0"

# Core exports (low-level primitives)
from .core import (
    SemanticSurface,
    RetrievalResult,
    HashIndex,
    MessageEntry,
)

# Component exports (high-level building blocks)
from .components import (
    # Semantic analysis
    analyze_relationships,
    format_for_llm,
    # Memory domains
    Domain,
    DomainType,
    DomainManager,
    DomainError,
    DomainNotFoundError,
    DomainAlreadyExistsError,
    DomainInactiveError,
    InvalidDomainPathError,
    # Baseline
    BaselineRetrieval,
    BaselineResult,
)

# Utils (internal, but exposed)
from .utils import PerformanceTracker, log_performance, track_performance

# Security utilities
from .security import check_prompt, SecurityCheck

# Memory harness (high-level interface)
from .memory_harness import MemoryHarness

# Utilities
from .utils.chunking import TextChunker, Chunk, chunk_by_chapters

# Integrations
from .integrations import (
    GutenbergText,
    fetch_gutenberg_text,
    load_gutenberg_text,
    get_recommended_work_id,
    list_recommended_works,
)

__all__ = [
    # Version
    '__version__',

    # Core primitives
    'SemanticSurface',
    'RetrievalResult',
    'HashIndex',
    'MessageEntry',

    # High-level interface
    'MemoryHarness',

    # Semantic analysis
    'analyze_relationships',
    'format_for_llm',

    # Memory domains
    'Domain',
    'DomainType',
    'DomainManager',
    'DomainError',
    'DomainNotFoundError',
    'DomainAlreadyExistsError',
    'DomainInactiveError',
    'InvalidDomainPathError',

    # Baseline
    'BaselineRetrieval',
    'BaselineResult',

    # Utils
    'PerformanceTracker',
    'log_performance',
    'track_performance',
    'TextChunker',
    'Chunk',
    'chunk_by_chapters',

    # Integrations
    'GutenbergText',
    'fetch_gutenberg_text',
    'load_gutenberg_text',
    'get_recommended_work_id',
    'list_recommended_works',

    # Security
    'check_prompt',
    'SecurityCheck',
]
