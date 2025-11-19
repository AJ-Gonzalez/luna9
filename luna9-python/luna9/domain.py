"""
Domain: Knowledge surface with metadata and hierarchy.

A Domain wraps a SemanticSurface with:
- Identity (name, path, type)
- Metadata (created_at, message_count, etc.)
- Lifecycle management (add, query, save/load)
"""

from typing import List, Dict, Optional, Any
from datetime import datetime
from pathlib import Path
from enum import Enum

from .semantic_surface import SemanticSurface, RetrievalResult


class DomainType(Enum):
    """Domain type classification."""
    PERSONAL = "personal"           # Relationship, communication, us
    FOUNDATION = "foundation"       # Reference knowledge (books, papers)
    PROJECT = "project"            # Dynamic project-specific domains
    CONVERSATION = "conversation"  # Active session memory
    HISTORY = "history"           # Archived conversations


class Domain:
    """
    A knowledge domain - semantic surface with identity and metadata.

    Represents a distinct area of knowledge with its own semantic surface.
    Domains can be hierarchical (e.g., foundation/books/rust_book).
    """

    def __init__(
        self,
        name: str,
        domain_type: DomainType,
        surface: Optional[SemanticSurface] = None,
        parent_path: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Create a new domain.

        Args:
            name: Domain name (e.g., "personal", "rust_book")
            domain_type: Type of domain
            surface: Existing SemanticSurface or None to create empty
            parent_path: Parent domain path (e.g., "foundation/books")
            metadata: Optional additional metadata
        """
        self.name = name
        self.domain_type = domain_type
        self.parent_path = parent_path
        self.created_at = datetime.now()
        self.last_modified = self.created_at
        self.metadata = metadata or {}

        # Construct full path
        if parent_path:
            self.path = f"{parent_path}/{name}"
        else:
            self.path = name

        # Initialize surface (can be empty)
        self.surface = surface
        self._message_metadata: List[Dict] = []  # Per-message metadata
        self._active: bool = True  # Domain is active by default

    @classmethod
    def create_empty(
        cls,
        name: str,
        domain_type: DomainType,
        parent_path: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> 'Domain':
        """
        Create an empty domain (no initial messages).

        Args:
            name: Domain name
            domain_type: Type of domain
            parent_path: Parent domain path
            metadata: Optional metadata

        Returns:
            Empty Domain
        """
        return cls(
            name=name,
            domain_type=domain_type,
            surface=None,
            parent_path=parent_path,
            metadata=metadata
        )

    @classmethod
    def create_from_messages(
        cls,
        name: str,
        domain_type: DomainType,
        messages: List[str],
        parent_path: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        message_metadata: Optional[List[Dict]] = None
    ) -> 'Domain':
        """
        Create domain from initial messages.

        Args:
            name: Domain name
            domain_type: Type of domain
            messages: Initial messages
            parent_path: Parent domain path
            metadata: Optional domain metadata
            message_metadata: Optional per-message metadata

        Returns:
            Domain with populated surface
        """
        surface = SemanticSurface(messages)
        domain = cls(
            name=name,
            domain_type=domain_type,
            surface=surface,
            parent_path=parent_path,
            metadata=metadata
        )

        if message_metadata:
            domain._message_metadata = message_metadata

        return domain

    def add_message(
        self,
        message: str,
        metadata: Optional[Dict] = None,
        rebuild_threshold: float = 0.1
    ) -> None:
        """
        Add a single message to the domain.

        Args:
            message: Text message
            metadata: Optional message metadata
            rebuild_threshold: Rebuild surface when pending >= threshold * size
        """
        # Create surface if this is first message
        if self.surface is None:
            self.surface = SemanticSurface([message])
            self._message_metadata = [metadata or {}]
        else:
            self.surface.append_message(message, metadata, rebuild_threshold)
            self._message_metadata.append(metadata or {})

        self.last_modified = datetime.now()

    def add_messages(
        self,
        messages: List[str],
        metadata: Optional[List[Dict]] = None,
        rebuild_threshold: float = 0.1
    ) -> None:
        """
        Add multiple messages to the domain (batch).

        Args:
            messages: List of text messages
            metadata: Optional list of per-message metadata
            rebuild_threshold: Rebuild surface when pending >= threshold * size
        """
        if metadata is None:
            metadata = [None] * len(messages)

        # Create surface if this is first batch
        if self.surface is None:
            self.surface = SemanticSurface(messages)
            self._message_metadata = [m or {} for m in metadata]
        else:
            self.surface.append_messages(messages, metadata, rebuild_threshold)
            self._message_metadata.extend([m or {} for m in metadata])

        self.last_modified = datetime.now()

    def query(
        self,
        query_text: str,
        k: int = 5,
        mode: str = 'both'
    ) -> Dict[str, Any]:
        """
        Query the domain's semantic surface.

        Args:
            query_text: Query string
            k: Number of results
            mode: 'smooth', 'exact', or 'both'

        Returns:
            Dict with results and domain context
        """
        if self.surface is None:
            return {
                'domain': self.path,
                'results': [],
                'message': 'Domain is empty'
            }

        result = self.surface.query(query_text, k=k)
        messages = result.get_messages(self.surface.messages, mode=mode, k=k)

        # Add domain context
        return {
            'domain': self.path,
            'domain_type': self.domain_type.value,
            'uv': result.uv,
            'curvature': result.curvature,
            **messages
        }

    def get_info(self) -> Dict[str, Any]:
        """
        Get domain information and statistics.

        Returns:
            Dict with domain metadata and stats
        """
        message_count = len(self.surface.messages) if self.surface else 0
        grid_size = f"{self.surface.grid_m}x{self.surface.grid_n}" if self.surface else "N/A"

        return {
            'name': self.name,
            'path': self.path,
            'type': self.domain_type.value,
            'created_at': self.created_at.isoformat(),
            'last_modified': self.last_modified.isoformat(),
            'message_count': message_count,
            'grid_size': grid_size,
            'metadata': self.metadata
        }

    def __repr__(self) -> str:
        msg_count = len(self.surface.messages) if self.surface else 0
        return f"Domain(path='{self.path}', type={self.domain_type.value}, messages={msg_count})"
