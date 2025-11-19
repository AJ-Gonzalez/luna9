"""
Luna Nine - Geometric Memory for Language Models

Navigate semantic space through parametric surfaces.
"""

__version__ = "0.1.0"

from .semantic_surface import SemanticSurface, RetrievalResult
from .baseline import BaselineRetrieval, BaselineResult
from .domain import Domain, DomainType
from .domain_manager import (
    DomainManager,
    DomainError,
    DomainNotFoundError,
    DomainAlreadyExistsError,
    DomainInactiveError,
    InvalidDomainPathError
)

__all__ = [
    'SemanticSurface',
    'RetrievalResult',
    'BaselineRetrieval',
    'BaselineResult',
    'Domain',
    'DomainType',
    'DomainManager',
    'DomainError',
    'DomainNotFoundError',
    'DomainAlreadyExistsError',
    'DomainInactiveError',
    'InvalidDomainPathError'
]
