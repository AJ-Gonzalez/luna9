"""
DomainManager: Orchestrator for hierarchical geometric memory domains.

Provides LLM agency over memory through tool interface for discovery, search,
mutation, and lifecycle management of semantic surface domains.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime

import numpy as np

from .domain import Domain, DomainType
from .semantic_surface import SemanticSurface


# Exception hierarchy
class DomainError(Exception):
    """Base exception for domain operations"""


class DomainNotFoundError(DomainError):
    """Raised when trying to access a domain that doesn't exist"""


class DomainAlreadyExistsError(DomainError):
    """Raised when trying to create a domain that already exists"""


class DomainInactiveError(DomainError):
    """Raised when trying to use an unloaded/inactive domain"""


class InvalidDomainPathError(DomainError):
    """Raised when domain path is invalid (too deep, invalid format, etc.)"""


class DomainManager:
    """
    Manages hierarchical domains and provides tool interface for LLM agency.

    Domains are organized in slash-separated hierarchies (max 3 levels):
        personal
        foundation/books/rust_book
        projects/luna_nine
        conversation/current_session

    The DomainManager enables multi-turn iterative exploration through simple
    tools that compose powerfully.
    """

    MAX_DEPTH = 3

    def __init__(self, storage_dir: Optional[Path] = None):
        """
        Initialize domain manager.

        Args:
            storage_dir: Base directory for persistent storage.
                        Defaults to ~/.luna9/domains/
                        TODO: Support LUNA9_DATA_DIR environment variable override
        """
        self.domains: Dict[str, Domain] = {}

        # Set up storage directory
        if storage_dir is None:
            self.storage_dir = Path.home() / ".luna9" / "domains"
        else:
            self.storage_dir = Path(storage_dir)

        # Create storage directory if it doesn't exist
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    def _get_domain_storage_path(self, domain_path: str) -> Path:
        """
        Get filesystem path for domain storage.

        Args:
            domain_path: Domain path like "foundation/books/rust"

        Returns:
            Path object for domain storage directory
        """
        # Convert forward-slash domain path to OS-appropriate path
        path_parts = domain_path.split('/')
        return self.storage_dir.joinpath(*path_parts)

    def _parse_path(self, path: str) -> Tuple[str, ...]:
        """
        Parse domain path into tuple of components.

        Args:
            path: Slash-separated path like "foundation/books/rust_book"

        Returns:
            Tuple of path components: ("foundation", "books", "rust_book")
        """
        return tuple(path.strip('/').split('/'))

    def _validate_path(self, path: str) -> None:
        """
        Validate domain path format and depth.

        Args:
            path: Domain path to validate

        Raises:
            InvalidDomainPathError: If path is invalid
        """
        if not path or path.strip() == '':
            raise InvalidDomainPathError("Domain path cannot be empty")

        # Check for invalid characters
        if any(c in path for c in ['\\', ' ', '\t', '\n']):
            raise InvalidDomainPathError(
                f"Domain path contains invalid characters: '{path}'"
            )

        # Check depth
        components = self._parse_path(path)
        if len(components) > self.MAX_DEPTH:
            raise InvalidDomainPathError(
                f"Domain path too deep (max {self.MAX_DEPTH} levels): '{path}'"
            )

        # Check for empty components
        if any(not c for c in components):
            raise InvalidDomainPathError(
                f"Domain path has empty components: '{path}'"
            )

    def _find_domain(self, path: str, require_active: bool = True) -> Optional[Domain]:
        """
        Find domain by path.

        Args:
            path: Domain path to find
            require_active: If True, raise error if domain is inactive

        Returns:
            Domain object if found, None otherwise

        Raises:
            DomainInactiveError: If domain exists but is inactive and require_active=True
        """
        domain = self.domains.get(path)

        if domain and require_active and not domain._active:
            raise DomainInactiveError(
                f"Domain '{path}' is inactive. Use load_domain() to activate."
            )

        return domain

    def _list_children(self, parent_path: Optional[str]) -> List[str]:
        """
        Find immediate child domains under a parent path (not grandchildren).

        Args:
            parent_path: Parent path to search under, or None for root

        Returns:
            List of immediate child domain paths
        """
        if parent_path is None:
            # Root level - find all domains with only 1 component
            return [
                path for path in self.domains.keys()
                if self.domains[path]._active and len(self._parse_path(path)) == 1
            ]

        # Find immediate children only (depth = parent_depth + 1)
        parent_components = self._parse_path(parent_path)
        parent_depth = len(parent_components)

        children = []
        for path in self.domains.keys():
            if not self.domains[path]._active:
                continue

            components = self._parse_path(path)

            # Check if this is an immediate child (exactly 1 level deeper)
            if (len(components) == parent_depth + 1 and
                components[:parent_depth] == parent_components):
                children.append(path)

        return children

    def _list_all_descendants(self, parent_path: str) -> List[str]:
        """
        Find ALL descendant domains under a parent path (children, grandchildren, etc).

        Args:
            parent_path: Parent path to search under

        Returns:
            List of all descendant domain paths
        """
        parent_components = self._parse_path(parent_path)
        parent_depth = len(parent_components)

        descendants = []
        for path in self.domains.keys():
            if not self.domains[path]._active:
                continue

            components = self._parse_path(path)

            # Check if this is a descendant of parent_path
            if (len(components) > parent_depth and
                components[:parent_depth] == parent_components):
                descendants.append(path)

        return descendants

    # -------------------------------------------------------------------------
    # Discovery
    # -------------------------------------------------------------------------

    def list_domains(self, path: Optional[str] = None) -> List[str]:
        """
        List domains at a given path level.

        Args:
            path: Parent path to list under, or None for root level

        Returns:
            List of domain paths

        Examples:
            list_domains() → ["personal", "foundation", "projects"]
            list_domains("foundation") → ["foundation/books", "foundation/papers"]
        """
        if path is not None:
            self._validate_path(path)

        if path is None:
            # Root level
            return sorted(self._list_children(None))

        # Check if there are children at this path
        children = self._list_children(path)
        if children:
            # This path has children, return them
            return sorted(children)

        # No children - check if this is a leaf domain
        domain = self._find_domain(path, require_active=True)
        if domain:
            # This is a leaf domain, return just this path
            return [path]

        # Path doesn't exist at all
        raise DomainNotFoundError(
            f"No domains found at path '{path}'. "
            f"Available domains: {list(self.domains.keys())}"
        )

    def get_domain_info(self, path: str) -> Dict[str, Any]:
        """
        Get information about a domain.

        Args:
            path: Domain path

        Returns:
            Dict with domain metadata and statistics

        Raises:
            DomainNotFoundError: If domain doesn't exist
        """
        self._validate_path(path)
        domain = self._find_domain(path, require_active=True)

        if not domain:
            available = list(self.domains.keys())
            raise DomainNotFoundError(
                f"Domain '{path}' not found. Available domains: {available}"
            )

        return domain.get_info()

    # -------------------------------------------------------------------------
    # Search
    # -------------------------------------------------------------------------

    def search_domain(
        self,
        path: str,
        query: str,
        k: int = 5,
        mode: str = "semantic"
    ) -> List[Dict[str, Any]]:
        """
        Search domain(s) for query.

        If path is a leaf domain, searches that domain.
        If path is a parent, searches all child domains and merges results.

        Args:
            path: Domain path (leaf or parent)
            query: Query text
            k: Number of results to return
            mode: Search mode - "semantic" | "literal" | "both"

        Returns:
            List of result dicts with structure:
                {
                    "domain_path": str,
                    "text": str,
                    "score": float,
                    "metadata": dict,
                    # If mode is "semantic" or "both":
                    "uv": (float, float),
                    "provenance": {"point_indices": [...], "weights": [...]}
                }

        Raises:
            DomainNotFoundError: If no domains found at path
            ValueError: If mode is invalid
        """
        self._validate_path(path)

        if mode not in ("semantic", "literal", "both"):
            raise ValueError(
                f"Invalid mode '{mode}'. Must be 'semantic', 'literal', or 'both'"
            )

        # Check if this is a leaf domain with content
        domain = self._find_domain(path, require_active=True)

        # If domain exists and has a surface (messages), search it directly
        # Otherwise, search all descendants
        if domain and domain.surface:
            # Search single domain with content
            results = self._search_single_domain(domain, query, k, mode)
        else:
            # Search all descendants (or it's an empty parent domain)
            descendants = self._list_all_descendants(path)
            if not descendants:
                # No descendants and either no domain or empty domain
                if domain:
                    # Empty domain, no results
                    return []
                else:
                    # Path doesn't exist
                    raise DomainNotFoundError(
                        f"No domains found at path '{path}'. "
                        f"Available domains: {list(self.domains.keys())}"
                    )

            # Search each descendant and merge
            all_results = []
            for descendant_path in descendants:
                descendant_domain = self._find_domain(descendant_path, require_active=True)
                # Only search domains that have content
                if descendant_domain and descendant_domain.surface:
                    descendant_results = self._search_single_domain(
                        descendant_domain, query, k, mode
                    )
                    all_results.extend(descendant_results)

            # Sort by score and limit to k
            results = sorted(all_results, key=lambda r: r['score'])[:k]

        return results

    def _search_single_domain(
        self,
        domain: Domain,
        query: str,
        k: int,
        mode: str
    ) -> List[Dict[str, Any]]:
        """
        Execute search against a single domain and format results for API.

        This is the core search implementation that bridges Domain's internal
        query method with DomainManager's standardized result format. It handles
        both exact nearest-neighbor lookup and smooth surface interpolation modes.

        Search strategy:
        1. Query domain with mode='both' to get complete result set
        2. Extract 'sources' (exact nearest neighbors from control points)
        3. Extract 'interpretation' (smooth surface interpolation)
        4. Format each result with domain context and metadata
        5. Include geometric data (UV coords, provenance) for semantic modes

        Result format per message:
        - domain_path: Full hierarchical path to source domain
        - text: Retrieved message text
        - score: Distance/similarity score (lower = more similar)
        - metadata: Message-level metadata (timestamps, speaker, etc.)
        - uv: Surface coordinates (only in semantic/both modes)
        - provenance: Influence weights from control points (semantic/both only)

        Args:
            domain: Domain instance to search
            query: Query text string
            k: Number of results to return
            mode: 'literal' (exact only), 'semantic' (surface only), 'both'

        Returns:
            List of formatted result dicts, one per retrieved message
        """
        # Use Domain's query method
        # Domain.query() returns 'interpretation' (weighted interpolation) and 'sources' (nearest neighbors)
        raw_results = domain.query(query, k=k, mode='both')

        # Use 'sources' which has the nearest neighbors with distances
        sources = raw_results.get('sources', {})
        messages = sources.get('messages', [])
        distances = sources.get('distances', [])
        indices = sources.get('indices', [])

        formatted_results = []
        for i, (text, distance, idx) in enumerate(zip(messages, distances, indices)):
            formatted = {
                'domain_path': domain.path,
                'text': text,
                'score': float(distance),  # Convert numpy to Python float
                'metadata': {}  # TODO: Add message metadata if available
            }

            # Add geometric data for semantic/both modes
            if mode in ('semantic', 'both'):
                formatted['uv'] = raw_results.get('uv', (0.0, 0.0))
                # Provenance from interpretation
                interpretation = raw_results.get('interpretation', {})
                formatted['provenance'] = {
                    'point_indices': interpretation.get('indices', []),
                    'weights': [float(w) for w in interpretation.get('weights', [])]
                }

            formatted_results.append(formatted)

        return formatted_results

    # -------------------------------------------------------------------------
    # Mutation
    # -------------------------------------------------------------------------

    def create_domain(
        self,
        path: str,
        domain_type: str
    ) -> Dict[str, Any]:
        """
        Create a new empty domain.

        Args:
            path: Domain path (must not exist)
            domain_type: Domain type string (PERSONAL, FOUNDATION, PROJECT, etc.)

        Returns:
            Dict with confirmation

        Raises:
            DomainAlreadyExistsError: If domain already exists
            InvalidDomainPathError: If path is invalid
        """
        self._validate_path(path)

        if path in self.domains:
            raise DomainAlreadyExistsError(
                f"Domain '{path}' already exists"
            )

        # Parse domain type
        try:
            dtype = DomainType[domain_type.upper()]
        except KeyError:
            valid_types = [t.name for t in DomainType]
            raise ValueError(
                f"Invalid domain type '{domain_type}'. "
                f"Must be one of: {valid_types}"
            )

        # Parse parent path
        components = self._parse_path(path)
        if len(components) > 1:
            parent_path = '/'.join(components[:-1])
        else:
            parent_path = None

        # Create domain
        domain = Domain.create_empty(
            name=components[-1],
            domain_type=dtype,
            parent_path=parent_path
        )
        domain._active = True

        self.domains[path] = domain

        return {
            'status': 'created',
            'path': path,
            'domain_type': domain_type
        }

    def add_to_domain(
        self,
        path: str,
        messages: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """
        Add messages to existing domain.

        Args:
            path: Domain path
            messages: List of message texts to add
            metadata: Optional list of metadata dicts, one per message.
                     Supports source attribution for provenance tracking:
                     {
                         "speaker": "user" | "agent" | "system",
                         "source_type": "original" | "quoted" | "external",
                         "source_attribution": {
                             "type": "book" | "article" | "documentation" | "code",
                             "title": str,
                             "author": str,
                             "page": str,
                             "exact_quote": bool
                         }
                     }

        Note: Source attribution enables fact-checking and citation verification.
              Client implementations should separate messages by author/source.

        Raises:
            DomainNotFoundError: If domain doesn't exist
        """
        self._validate_path(path)
        domain = self._find_domain(path, require_active=True)

        if not domain:
            raise DomainNotFoundError(
                f"Domain '{path}' not found. Use create_domain() first."
            )

        # Add messages
        if metadata:
            for msg, meta in zip(messages, metadata):
                domain.add_message(msg, meta)
        else:
            domain.add_messages(messages)

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------

    def save_domain(self, path: str) -> Dict[str, Any]:
        """
        Save domain to persistent storage.

        Saves domain metadata as JSON and surface data as .npz (numpy compressed).

        Storage structure:
            ~/.luna9/domains/{hierarchical/path}/
                domain.json    - Metadata, messages, timestamps
                surface.npz    - Embeddings, control points, knot vectors

        Args:
            path: Domain path to save

        Returns:
            Dict with save status and file paths

        Raises:
            DomainNotFoundError: If domain doesn't exist
        """
        self._validate_path(path)
        domain = self._find_domain(path, require_active=False)

        if not domain:
            raise DomainNotFoundError(
                f"Domain '{path}' not found"
            )

        # Get storage path and create directory
        storage_path = self._get_domain_storage_path(path)
        storage_path.mkdir(parents=True, exist_ok=True)

        # Prepare metadata for JSON
        domain_json = {
            'format_version': '1.0',
            'name': domain.name,
            'path': domain.path,
            'domain_type': domain.domain_type.value,
            'parent_path': domain.parent_path,
            'created_at': domain.created_at.isoformat(),
            'last_modified': domain.last_modified.isoformat(),
            'metadata': domain.metadata,
            'active': domain._active,
            'messages': domain.surface.messages if domain.surface else [],
            'message_metadata': domain._message_metadata,
            'model_name': domain.surface.model_name if domain.surface else None,
            'grid_shape': [domain.surface.grid_m, domain.surface.grid_n] if domain.surface else None
        }

        # Save metadata as JSON
        json_path = storage_path / 'domain.json'
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(domain_json, f, indent=2, ensure_ascii=False)

        # Save surface data if it exists
        npz_path = None
        if domain.surface:
            npz_path = storage_path / 'surface.npz'
            surface_data = {
                'embeddings': domain.surface.embeddings,
                'control_points': domain.surface.control_points,
                'weights': domain.surface.weights
            }
            np.savez_compressed(npz_path, **surface_data)

        # Save hash index if it exists
        hash_index_path = None
        if domain.hash_index:
            hash_index_path = storage_path / 'hash_index.pkl'
            domain.hash_index.save(hash_index_path)

        return {
            'status': 'saved',
            'path': path,
            'json_path': str(json_path),
            'npz_path': str(npz_path) if npz_path else None,
            'hash_index_path': str(hash_index_path) if hash_index_path else None
        }

    def load_domain(self, path: str) -> Dict[str, str]:
        """
        Load domain from persistent storage into active memory.

        Lazy loads domain on first access. If domain is already in memory,
        marks it as active.

        Args:
            path: Domain path to load

        Returns:
            Status dict

        Raises:
            DomainNotFoundError: If domain doesn't exist on disk or in memory
        """
        self._validate_path(path)

        # Check if already in memory
        domain = self._find_domain(path, require_active=False)
        if domain:
            if not domain._active:
                domain._active = True
                return {'status': 'activated', 'path': path}
            return {'status': 'already_loaded', 'path': path}

        # Try to load from disk
        storage_path = self._get_domain_storage_path(path)
        json_path = storage_path / 'domain.json'

        if not json_path.exists():
            raise DomainNotFoundError(
                f"Domain '{path}' not found on disk or in memory. "
                f"Looked for: {json_path}"
            )

        # Load metadata from JSON
        with open(json_path, 'r', encoding='utf-8') as f:
            domain_json = json.load(f)

        # Validate format version
        format_version = domain_json.get('format_version', '1.0')
        if format_version != '1.0':
            raise ValueError(
                f"Unsupported domain format version: {format_version}. "
                f"Expected 1.0"
            )

        # Load surface data if it exists
        npz_path = storage_path / 'surface.npz'
        surface = None
        if npz_path.exists() and domain_json.get('messages'):
            # Load surface arrays
            surface_data = np.load(npz_path)

            # Get grid shape and model name from JSON
            grid_shape = tuple(domain_json['grid_shape'])
            model_name = domain_json.get('model_name', 'all-MiniLM-L6-v2')

            # Reconstruct SemanticSurface with pre-computed embeddings
            messages = domain_json['messages']
            embeddings = surface_data['embeddings']
            surface = SemanticSurface(
                messages,
                embeddings=embeddings,
                model_name=model_name,
                grid_shape=grid_shape
            )

            # Restore weights (control_points are set during __init__)
            surface.weights = surface_data['weights']

        # Reconstruct Domain
        domain_type = DomainType(domain_json['domain_type'])
        domain = Domain(
            name=domain_json['name'],
            domain_type=domain_type,
            surface=surface,
            parent_path=domain_json.get('parent_path'),
            metadata=domain_json.get('metadata', {})
        )

        # Restore timestamps
        domain.created_at = datetime.fromisoformat(domain_json['created_at'])
        domain.last_modified = datetime.fromisoformat(domain_json['last_modified'])
        domain._message_metadata = domain_json.get('message_metadata', [])
        domain._active = domain_json.get('active', True)

        # Load hash index if it exists
        hash_index_path = storage_path / 'hash_index.pkl'
        if hash_index_path.exists():
            from .hash_index import HashIndex
            domain.hash_index = HashIndex.load(hash_index_path)

        # Add to registry
        self.domains[path] = domain

        return {'status': 'loaded', 'path': path}

    def unload_domain(self, path: str, save_first: bool = True) -> Dict[str, Any]:
        """
        Unload domain from active memory.

        By default, saves to disk before unloading. The domain can be reloaded
        later with load_domain().

        Args:
            path: Domain path to unload
            save_first: If True, save to disk before unloading (default: True)

        Returns:
            Dict with status

        Raises:
            DomainNotFoundError: If domain doesn't exist
        """
        self._validate_path(path)
        domain = self._find_domain(path, require_active=False)

        if not domain:
            raise DomainNotFoundError(
                f"Domain '{path}' not found"
            )

        # Save before unloading if requested
        if save_first:
            self.save_domain(path)

        # Mark inactive and remove from active registry
        domain._active = False

        # TODO: In future, consider removing from self.domains entirely
        # to free memory. For now, keep in memory but mark inactive.
        # NOTE: Placeholder for future auto-save on unload

        return {
            'status': 'unloaded',
            'path': path,
            'saved': save_first
        }

    def delete_domain(self, path: str, confirm: bool = False) -> Dict[str, Any]:
        """
        Permanently delete domain from memory and disk.

        WARNING: This action cannot be undone. All data will be lost.

        Args:
            path: Domain path to delete
            confirm: Must be True to actually delete (safety check)

        Returns:
            Dict with status and what was deleted

        Raises:
            ValueError: If confirm=False (prevents accidents)
            DomainNotFoundError: If domain doesn't exist in memory or on disk
        """
        if not confirm:
            raise ValueError(
                f"Must set confirm=True to delete domain '{path}'. "
                "This action cannot be undone."
            )

        self._validate_path(path)
        domain = self._find_domain(path, require_active=False)

        # Get info before deleting
        message_count = 0
        if domain:
            message_count = len(domain.surface.messages) if domain.surface else 0
            # Remove from memory registry
            del self.domains[path]

        # Delete from disk if it exists
        storage_path = self._get_domain_storage_path(path)
        disk_deleted = False
        if storage_path.exists():
            import shutil
            shutil.rmtree(storage_path)
            disk_deleted = True

        if not domain and not disk_deleted:
            raise DomainNotFoundError(
                f"Domain '{path}' not found in memory or on disk"
            )

        return {
            'status': 'deleted',
            'path': path,
            'message_count': message_count,
            'disk_deleted': disk_deleted
        }
