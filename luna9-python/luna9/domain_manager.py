"""
DomainManager: Orchestrator for hierarchical geometric memory domains.

Provides LLM agency over memory through tool interface for discovery, search,
mutation, and lifecycle management of semantic surface domains.
"""

from typing import Dict, List, Optional, Tuple, Any
from .domain import Domain, DomainType


# Exception hierarchy
class DomainError(Exception):
    """Base exception for domain operations"""
    pass


class DomainNotFoundError(DomainError):
    """Raised when trying to access a domain that doesn't exist"""
    pass


class DomainAlreadyExistsError(DomainError):
    """Raised when trying to create a domain that already exists"""
    pass


class DomainInactiveError(DomainError):
    """Raised when trying to use an unloaded/inactive domain"""
    pass


class InvalidDomainPathError(DomainError):
    """Raised when domain path is invalid (too deep, invalid format, etc.)"""
    pass


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

    def __init__(self):
        """Initialize empty domain registry."""
        self.domains: Dict[str, Domain] = {}

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

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
        Search a single domain and format results.

        Args:
            domain: Domain to search
            query: Query text
            k: Number of results
            mode: Search mode

        Returns:
            List of formatted results
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
            metadata: Optional list of metadata dicts (one per message)

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

    def load_domain(self, path: str) -> Dict[str, str]:
        """
        Load domain into active memory.

        NOTE: Currently a no-op (all domains in-memory).
        TODO: Implement disk persistence and actual loading.

        Args:
            path: Domain path to load

        Returns:
            Status dict

        Raises:
            DomainNotFoundError: If domain doesn't exist
        """
        self._validate_path(path)
        domain = self._find_domain(path, require_active=False)

        if not domain:
            raise DomainNotFoundError(
                f"Domain '{path}' not found"
            )

        if not domain._active:
            domain._active = True
            return {'status': 'loaded', 'path': path}

        return {'status': 'already_loaded', 'path': path}

    def unload_domain(self, path: str) -> None:
        """
        Mark domain as inactive (won't appear in searches or listings).

        Domain remains in memory but is hidden.
        TODO: When we add persistence, this should save to disk and free memory.

        Args:
            path: Domain path to unload

        Raises:
            DomainNotFoundError: If domain doesn't exist
        """
        self._validate_path(path)
        domain = self._find_domain(path, require_active=False)

        if not domain:
            raise DomainNotFoundError(
                f"Domain '{path}' not found"
            )

        domain._active = False

    def delete_domain(self, path: str, confirm: bool = False) -> Dict[str, Any]:
        """
        Permanently delete domain from memory (and disk, when persistence added).

        Args:
            path: Domain path to delete
            confirm: Must be True to actually delete (safety check)

        Returns:
            Dict with status and what was deleted

        Raises:
            ValueError: If confirm=False (prevents accidents)
            DomainNotFoundError: If domain doesn't exist
        """
        if not confirm:
            raise ValueError(
                f"Must set confirm=True to delete domain '{path}'. "
                "This action cannot be undone."
            )

        self._validate_path(path)
        domain = self._find_domain(path, require_active=False)

        if not domain:
            raise DomainNotFoundError(
                f"Domain '{path}' not found"
            )

        # Get info before deleting
        message_count = len(domain.surface.messages) if domain.surface else 0

        # Remove from registry
        del self.domains[path]

        return {
            'status': 'deleted',
            'path': path,
            'message_count': message_count
        }
