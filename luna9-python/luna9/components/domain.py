"""
Domain: Knowledge surface with metadata and hierarchy.

A Domain wraps a SemanticSurface with:
- Identity (name, path, type)
- Metadata (created_at, message_count, etc.)
- Lifecycle management (add, query, save/load)
"""

from typing import List, Dict, Optional, Any
from datetime import datetime
from enum import Enum

from ..core.semantic_surface import SemanticSurface
from ..core.hash_index import HashIndex


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
        metadata: Optional[Dict[str, Any]] = None,
        use_hash_index: bool = True
    ):
        """
        Create a new domain.

        Args:
            name: Domain name (e.g., "personal", "rust_book")
            domain_type: Type of domain
            surface: Existing SemanticSurface or None to create empty
            parent_path: Parent domain path (e.g., "foundation/books")
            metadata: Optional additional metadata
            use_hash_index: Enable hash bucketing for O(1) lookups (default True)
        """
        self.name = name
        self.domain_type = domain_type
        self.parent_path = parent_path
        self.created_at = datetime.now()
        self.last_modified = self.created_at
        self.metadata = metadata or {}
        self.use_hash_index = use_hash_index

        # Construct full path
        if parent_path:
            self.path = f"{parent_path}/{name}"
        else:
            self.path = name

        # Initialize surface (can be empty)
        self.surface = surface
        self._message_metadata: List[Dict] = []  # Per-message metadata
        self._active: bool = True  # Domain is active by default

        # Initialize hash index for fast lookups
        # Only use hash index if surface is large enough to benefit
        # Hash index overhead isn't worth it for small grids (< 8x8 = 64 control points)
        # Benchmark shows ~10-40x speedup only applies when grid_size >= 8
        if use_hash_index and surface is not None:
            grid_size = max(surface.grid_m, surface.grid_n)
            if grid_size < 8:
                # Grid too small - full scan is faster than hash overhead
                self.hash_index = None
            else:
                self.hash_index = HashIndex()
        elif use_hash_index:
            # Surface doesn't exist yet, create hash index optimistically
            # Will be disabled on first add_messages if grid ends up too small
            self.hash_index = HashIndex()
        else:
            self.hash_index = None

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
        message_metadata: Optional[List[Dict]] = None,
        use_hash_index: bool = True
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
            use_hash_index: Enable hash bucketing for O(1) lookups (default True)

        Returns:
            Domain with populated surface
        """
        surface = SemanticSurface(messages)
        domain = cls(
            name=name,
            domain_type=domain_type,
            surface=surface,
            parent_path=parent_path,
            metadata=metadata,
            use_hash_index=use_hash_index
        )

        if message_metadata:
            domain._message_metadata = message_metadata

        # Populate hash index with initial messages
        if domain.hash_index is not None:
            # Project all messages and add to hash index
            for i in range(len(messages)):
                # Get embedding from surface (already computed)
                embedding = surface.embeddings[i]
                u, v = surface.project_embedding(embedding)
                domain.hash_index.add_message(i, u, v)

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
        # Track message index before adding
        message_idx = len(self.surface.messages) if self.surface else 0

        # Create surface if this is first message
        if self.surface is None:
            self.surface = SemanticSurface([message])
            self._message_metadata = [metadata or {}]

            # Check if grid is too small for hash index
            if self.hash_index is not None:
                grid_size = max(self.surface.grid_m, self.surface.grid_n)
                if grid_size < 8:
                    self.hash_index = None
        else:
            self.surface.append_message(message, metadata, rebuild_threshold)
            self._message_metadata.append(metadata or {})

        # Add to hash index if enabled
        if self.hash_index is not None:
            # Get embedding for this message
            model = self.surface._get_embedding_model()
            embedding = model.encode([message], show_progress_bar=False)[0]

            # Project to surface coordinates
            u, v = self.surface.project_embedding(embedding)

            # Add to hash index
            self.hash_index.add_message(message_idx, u, v)

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

        # Track starting index
        start_idx = len(self.surface.messages) if self.surface else 0

        # Create surface if this is first batch
        if self.surface is None:
            self.surface = SemanticSurface(messages)
            self._message_metadata = [m or {} for m in metadata]

            # Check if grid is too small for hash index to be beneficial
            if self.hash_index is not None:
                grid_size = max(self.surface.grid_m, self.surface.grid_n)
                if grid_size < 8:
                    # Disable hash index for small grids - full scan is faster
                    self.hash_index = None
        else:
            self.surface.append_messages(messages, metadata, rebuild_threshold)
            self._message_metadata.extend([m or {} for m in metadata])

            # Check if grid grew enough or is still too small for hash index
            if self.hash_index is not None:
                grid_size = max(self.surface.grid_m, self.surface.grid_n)
                if grid_size < 8:
                    # Still too small - disable hash index
                    self.hash_index = None

        # Add batch to hash index if enabled
        if self.hash_index is not None:
            # Get embeddings for all new messages
            model = self.surface._get_embedding_model()
            embeddings = model.encode(messages, show_progress_bar=False)

            # Project each to surface and add to hash index
            for i, embedding in enumerate(embeddings):
                message_idx = start_idx + i
                u, v = self.surface.project_embedding(embedding)
                self.hash_index.add_message(message_idx, u, v)

        self.last_modified = datetime.now()

    def _rebuild_hash_index(self) -> None:
        """
        Rebuild hash index from scratch after surface rebuild.

        Called when surface dimensions change due to lazy incremental updates.
        Clears existing hash index and repopulates from current surface state.
        """
        if self.hash_index is None:
            return

        # Clear existing entries (bucket size changes with new grid)
        from .hash_index import HashIndex
        self.hash_index = HashIndex(
            bucket_size=self.hash_index.bucket_size,
            quantization_bits=self.hash_index.quantization_bits
        )

        # Repopulate with all messages
        for i in range(len(self.surface.messages)):
            embedding = self.surface.embeddings[i]
            u, v = self.surface.project_embedding(embedding)
            self.hash_index.add_message(i, u, v)

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

        # Check if surface needs rebuild (happens lazily in surface.query)
        needs_rebuild = self.surface._dirty

        # Pass hash index to surface query for O(1) candidate retrieval
        result = self.surface.query(query_text, k=k, hash_index=self.hash_index)

        # Synchronize hash index after surface rebuild
        if needs_rebuild and self.hash_index is not None:
            self._rebuild_hash_index()
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
            'domain_path': self.path,
            'domain_type': self.domain_type.value,
            'created_at': self.created_at.isoformat(),
            'last_modified': self.last_modified.isoformat(),
            'message_count': message_count,
            'grid_size': grid_size,
            'metadata': self.metadata
        }

    def __repr__(self) -> str:
        msg_count = len(self.surface.messages) if self.surface else 0
        return f"Domain(path='{self.path}', type={self.domain_type.value}, messages={msg_count})"
