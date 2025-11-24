"""
Memory Harness - High-level interface for Luna 9 semantic memory.

Provides an intuitive API for continuous context management, document ingestion,
and intelligent recall across geometric memory surfaces.
"""

from typing import List, Dict, Any, Optional, Set, Literal
from pathlib import Path
import logging

from .components.domain_manager import DomainManager, DomainType
from .utils.chunking import TextChunker, Chunk, chunk_by_chapters
from .integrations.gutenberg import (
    GutenbergText,
    fetch_gutenberg_text,
    load_gutenberg_text,
    get_domain_path_for_gutenberg
)


logger = logging.getLogger(__name__)


class MemoryHarness:
    """
    High-level interface for Luna 9 semantic memory.

    Manages document ingestion, intelligent recall, and domain organization
    with minimal configuration required.

    Example:
        ```python
        harness = MemoryHarness()

        # Ingest a document
        harness.ingest_gutenberg("frankenstein")

        # Remember conversation context
        harness.remember(
            "We should add authentication to the API",
            context="project/myapp"
        )

        # Recall relevant information
        results = harness.recall(
            "What did we decide about security?",
            k=5
        )
        ```
    """

    def __init__(
        self,
        base_path: Optional[str] = None,
        auto_activate: bool = True
    ):
        """
        Initialize memory harness.

        Args:
            base_path: Base directory for domain storage (default: ~/.luna9)
            auto_activate: Automatically activate domains when created
        """
        self.manager = DomainManager(base_path)
        self.active_domains: Set[str] = set()
        self.auto_activate = auto_activate

        # Default chunking strategy
        self._default_chunker = TextChunker(
            strategy="semantic",
            max_chunk_size=2000,
            min_chunk_size=20
        )

    def _domain_exists(self, path: str) -> bool:
        """Check if a domain exists."""
        return path in self.manager.domains

    def remember(
        self,
        text: str,
        context: str = "conversation",
        source: str = "user",
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Remember a piece of information in appropriate domain.

        Automatically creates domain if it doesn't exist.

        Args:
            text: Text to remember
            context: Context identifier (becomes domain path)
            source: Source of information ("user", "agent", "system")
            metadata: Additional metadata to attach
        """
        # Ensure domain exists
        if not self._domain_exists(context):
            # Infer domain type from context
            domain_type = self._infer_domain_type(context)
            self.manager.create_domain(context, domain_type.value)

            if self.auto_activate:
                self.active_domains.add(context)

        # Prepare metadata
        msg_metadata = {
            "source": source,
            **(metadata or {})
        }

        # Add to domain
        self.manager.add_to_domain(context, [text], [msg_metadata])

        logger.info(f"Remembered in {context}: {text[:50]}...")

    def recall(
        self,
        query: str,
        k: int = 5,
        domains: Optional[List[str]] = None,
        mode: Literal["semantic", "simple"] = "semantic"
    ) -> List[Dict[str, Any]]:
        """
        Recall relevant information from memory.

        Args:
            query: Query text
            k: Number of results to return
            domains: List of domain paths to search (None = search active domains)
            mode: Search mode ("semantic" for full provenance, "simple" for fast)

        Returns:
            List of result dicts with:
                - text: Retrieved text
                - score: Relevance score (distance)
                - domain_path: Source domain
                - metadata: Associated metadata
                - uv: Surface coordinates (semantic mode only)
                - provenance: Contributing points and weights (semantic mode only)
        """
        search_domains = domains or list(self.active_domains)

        if not search_domains:
            logger.warning("No active domains for recall")
            return []

        all_results = []

        # Map mode to DomainManager's expected values
        manager_mode = "semantic" if mode == "semantic" else "literal"

        for domain_path in search_domains:
            try:
                results = self.manager.search_domain(
                    domain_path,
                    query,
                    k=k,
                    mode=manager_mode
                )
                all_results.extend(results)
            except Exception as e:
                logger.error(f"Search failed for {domain_path}: {e}")
                continue

        # Sort by score (distance) and limit to k
        all_results.sort(key=lambda x: x['score'])
        return all_results[:k]

    def ingest_document(
        self,
        text: str,
        domain: str,
        title: Optional[str] = None,
        chunker: Optional[TextChunker] = None,
        base_metadata: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Ingest a document into a domain.

        Args:
            text: Document text
            domain: Target domain path
            title: Document title (for metadata)
            chunker: Custom chunker (default: semantic chunking)
            base_metadata: Metadata to attach to all chunks

        Returns:
            Number of chunks ingested
        """
        # Ensure domain exists
        if not self._domain_exists(domain):
            domain_type = self._infer_domain_type(domain)
            self.manager.create_domain(domain, domain_type.value)

            if self.auto_activate:
                self.active_domains.add(domain)

        # Prepare metadata
        meta = base_metadata or {}
        if title:
            meta['title'] = title

        # Chunk the document
        if chunker is None:
            chunker = self._default_chunker

        chunks = chunker.chunk(text, meta)

        # Add chunks to domain
        messages = [chunk.text for chunk in chunks]
        metadata_list = [chunk.metadata for chunk in chunks]

        self.manager.add_to_domain(domain, messages, metadata_list)

        logger.info(f"Ingested {len(chunks)} chunks into {domain}")

        return len(chunks)

    def ingest_gutenberg(
        self,
        work_identifier: str | int,
        domain: Optional[str] = None,
        chunk_by_chapter: bool = True,
        chunker: Optional[TextChunker] = None
    ) -> Dict[str, Any]:
        """
        Ingest a Project Gutenberg work.

        Args:
            work_identifier: Gutenberg ID (int) or slug from RECOMMENDED_WORKS
            domain: Target domain (auto-generated if None)
            chunk_by_chapter: If True, chunk by chapters then paragraphs
            chunker: Custom chunker for chapter content

        Returns:
            Dict with ingestion info:
                - domain: Domain path used
                - chunks: Number of chunks ingested
                - metadata: Work metadata

        Example:
            ```python
            harness.ingest_gutenberg("frankenstein")
            harness.ingest_gutenberg(84)  # Same work by ID
            ```
        """
        # Resolve work identifier
        if isinstance(work_identifier, str):
            from .integrations.gutenberg import get_recommended_work_id
            gutenberg_id = get_recommended_work_id(work_identifier)
            if gutenberg_id is None:
                raise ValueError(f"Unknown work slug: {work_identifier}")
        else:
            gutenberg_id = work_identifier

        # Fetch the work
        logger.info(f"Fetching Gutenberg ID {gutenberg_id}...")
        work = fetch_gutenberg_text(gutenberg_id)

        # Determine domain path
        if domain is None:
            domain = get_domain_path_for_gutenberg(work)

        # Ensure domain exists
        if not self._domain_exists(domain):
            self.manager.create_domain(domain, DomainType.FOUNDATION.value)

            if self.auto_activate:
                self.active_domains.add(domain)

        # Chunk and ingest
        if chunk_by_chapter:
            if chunker is None:
                chunker = TextChunker(
                    strategy="semantic",
                    max_chunk_size=1500,
                    min_chunk_size=20
                )

            chunks = chunk_by_chapters(
                work.text,
                base_metadata=work.metadata,
                chunk_chapters=True,
                chunker=chunker
            )
        else:
            if chunker is None:
                chunker = self._default_chunker

            chunks = chunker.chunk(work.text, work.metadata)

        # Add to domain
        messages = [chunk.text for chunk in chunks]
        metadata_list = [chunk.metadata for chunk in chunks]

        self.manager.add_to_domain(domain, messages, metadata_list)

        logger.info(
            f"Ingested '{work.title}' ({len(chunks)} chunks) into {domain}"
        )

        return {
            'domain': domain,
            'chunks': len(chunks),
            'metadata': work.metadata
        }

    def ingest_local_file(
        self,
        file_path: str,
        domain: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Ingest a local text file.

        Supports Project Gutenberg .txt files (auto-detects metadata).

        Args:
            file_path: Path to file
            domain: Target domain (auto-generated from filename if None)
            **kwargs: Additional args passed to ingest_document

        Returns:
            Dict with ingestion info
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Try loading as Gutenberg text first
        try:
            work = load_gutenberg_text(file_path)

            if domain is None:
                domain = get_domain_path_for_gutenberg(work)

            # Use Gutenberg ingestion path
            return self.ingest_document(
                work.text,
                domain,
                title=work.title,
                base_metadata=work.metadata,
                **kwargs
            )

        except Exception:
            # Fall back to plain text
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()

            if domain is None:
                # Use filename as domain
                domain = f"documents/{path.stem}"

            chunk_count = self.ingest_document(
                text,
                domain,
                title=path.name,
                base_metadata={'filename': path.name},
                **kwargs
            )

            return {
                'domain': domain,
                'chunks': chunk_count,
                'metadata': {'filename': path.name}
            }

    def activate_domain(self, domain_path: str) -> None:
        """
        Add domain to active set for recall queries.

        Args:
            domain_path: Domain path to activate
        """
        if not self._domain_exists(domain_path):
            raise ValueError(f"Domain does not exist: {domain_path}")

        self.active_domains.add(domain_path)
        logger.info(f"Activated domain: {domain_path}")

    def deactivate_domain(self, domain_path: str) -> None:
        """
        Remove domain from active set.

        Args:
            domain_path: Domain path to deactivate
        """
        self.active_domains.discard(domain_path)
        logger.info(f"Deactivated domain: {domain_path}")

    def list_domains(self) -> List[str]:
        """List all available domains."""
        return self.manager.list_domains()

    def get_domain_info(self, domain_path: str) -> Dict[str, Any]:
        """
        Get information about a domain.

        Args:
            domain_path: Domain path

        Returns:
            Dict with domain metadata
        """
        return self.manager.get_domain_info(domain_path)

    def clear_domain(self, domain_path: str) -> None:
        """
        Delete a domain and all its data.

        Args:
            domain_path: Domain path to delete
        """
        self.active_domains.discard(domain_path)
        self.manager.delete_domain(domain_path, confirm=True)
        logger.info(f"Deleted domain: {domain_path}")

    def _infer_domain_type(self, domain_path: str) -> DomainType:
        """Infer domain type from path."""
        path_lower = domain_path.lower()

        if path_lower.startswith("foundation"):
            return DomainType.FOUNDATION
        elif path_lower.startswith("project"):
            return DomainType.PROJECT
        elif path_lower.startswith("conversation"):
            return DomainType.CONVERSATION
        elif path_lower.startswith("personal"):
            return DomainType.PERSONAL
        elif path_lower.startswith("history"):
            return DomainType.HISTORY
        else:
            # Default to conversation for ad-hoc contexts
            return DomainType.CONVERSATION

    @property
    def active_domain_count(self) -> int:
        """Number of currently active domains."""
        return len(self.active_domains)

    def __repr__(self) -> str:
        return (
            f"<MemoryHarness: "
            f"{len(self.manager.list_domains())} domains, "
            f"{self.active_domain_count} active>"
        )
