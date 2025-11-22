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

__all__ = [
    # Version
    '__version__',

    # Core primitives
    'SemanticSurface',
    'RetrievalResult',
    'HashIndex',
    'MessageEntry',

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
]
