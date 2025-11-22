"""
Luna Nine Components - High-level building blocks

This module contains user-facing components for common use cases:
- Semantic analysis (product positioning, relationship inference)
- Memory domains (conversation/project storage)
- Baseline retrieval (simple vector search for comparison)
"""

from .semantic_analysis import analyze_relationships, format_for_llm
from .domain import Domain, DomainType
from .domain_manager import (
    DomainManager,
    DomainError,
    DomainNotFoundError,
    DomainAlreadyExistsError,
    DomainInactiveError,
    InvalidDomainPathError
)
from .baseline import BaselineRetrieval, BaselineResult

__all__ = [
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
]
