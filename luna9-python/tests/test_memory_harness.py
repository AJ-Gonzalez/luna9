"""Tests for MemoryHarness high-level API."""

import pytest
import tempfile
import shutil
from pathlib import Path

from luna9 import MemoryHarness
from luna9.components.domain_manager import DomainType


@pytest.fixture
def temp_harness():
    """Create a temporary memory harness for testing."""
    temp_dir = tempfile.mkdtemp()
    harness = MemoryHarness(base_path=temp_dir, auto_activate=True)

    yield harness

    # Cleanup
    shutil.rmtree(temp_dir)


class TestMemoryHarnessBasics:
    """Basic memory harness functionality tests."""

    def test_initialization(self, temp_harness):
        """Test harness initializes correctly."""
        assert temp_harness is not None
        assert temp_harness.manager is not None
        assert temp_harness.active_domain_count == 0

    def test_remember_creates_domain(self, temp_harness):
        """Test that remember() creates domain automatically."""
        temp_harness.remember(
            "This is a test message",
            context="test/domain"
        )

        assert "test/domain" in temp_harness.manager.domains
        assert "test/domain" in temp_harness.active_domains

    def test_remember_and_recall(self, temp_harness):
        """Test basic remember and recall flow."""
        # Remember some information
        temp_harness.remember(
            "Luna 9 uses geometric surfaces for memory",
            context="conversation/test"
        )

        temp_harness.remember(
            "Semantic search is very efficient",
            context="conversation/test"
        )

        # Recall
        results = temp_harness.recall("geometric memory", k=5)

        assert len(results) > 0
        assert any("geometric" in r['text'].lower() for r in results)

    def test_domain_type_inference(self, temp_harness):
        """Test that domain types are inferred correctly from paths."""
        test_cases = [
            ("foundation/test", DomainType.FOUNDATION),
            ("project/myapp", DomainType.PROJECT),
            ("conversation/chat", DomainType.CONVERSATION),
            ("personal/notes", DomainType.PERSONAL),
            ("history/archive", DomainType.HISTORY),
            ("random/path", DomainType.CONVERSATION),  # Default
        ]

        for path, expected_type in test_cases:
            temp_harness.remember("test", context=path)
            info = temp_harness.get_domain_info(path)
            assert info['domain_type'] == expected_type.value

    def test_metadata_preservation(self, temp_harness):
        """Test that metadata is preserved through remember()."""
        temp_harness.remember(
            "Important message",
            context="test/meta",
            source="user",
            metadata={"priority": "high", "tag": "important"}
        )

        results = temp_harness.recall("important", k=1)
        assert len(results) > 0
        assert results[0]['metadata']['source'] == 'user'
        assert results[0]['metadata']['priority'] == 'high'


class TestDocumentIngestion:
    """Test document ingestion functionality."""

    def test_ingest_document_basic(self, temp_harness):
        """Test basic document ingestion."""
        text = """
This is a test document.

It has multiple paragraphs.

Each paragraph should become a chunk.
        """.strip()

        chunk_count = temp_harness.ingest_document(
            text,
            domain="test/doc",
            title="Test Document"
        )

        assert chunk_count > 0
        assert "test/doc" in temp_harness.manager.domains

        # Check domain has messages
        info = temp_harness.get_domain_info("test/doc")
        assert info['message_count'] == chunk_count

    def test_ingest_document_with_metadata(self, temp_harness):
        """Test document ingestion preserves metadata."""
        text = "Test content."

        temp_harness.ingest_document(
            text,
            domain="test/doc_meta",
            title="My Document",
            base_metadata={"author": "Test Author", "year": "2025"}
        )

        results = temp_harness.recall("test content", k=1)
        assert len(results) > 0
        assert results[0]['metadata']['title'] == 'My Document'
        assert results[0]['metadata']['author'] == 'Test Author'

    def test_ingest_with_custom_chunker(self, temp_harness):
        """Test ingestion with custom chunker."""
        from luna9 import TextChunker

        text = "A" * 1000  # Long text

        chunker = TextChunker(strategy="fixed", max_chunk_size=100)

        chunk_count = temp_harness.ingest_document(
            text,
            domain="test/custom",
            chunker=chunker
        )

        # Should create multiple fixed-size chunks
        assert chunk_count > 5


class TestDomainManagement:
    """Test domain activation and management."""

    def test_activate_deactivate(self, temp_harness):
        """Test activating and deactivating domains."""
        temp_harness.remember("test", context="domain1")
        temp_harness.remember("test", context="domain2")

        # Both should be active (auto_activate=True)
        assert temp_harness.active_domain_count == 2

        # Deactivate one
        temp_harness.deactivate_domain("domain1")
        assert temp_harness.active_domain_count == 1
        assert "domain1" not in temp_harness.active_domains

        # Reactivate
        temp_harness.activate_domain("domain1")
        assert temp_harness.active_domain_count == 2

    def test_recall_respects_active_domains(self, temp_harness):
        """Test that recall only searches active domains."""
        # Create two domains with different content
        temp_harness.remember("cats are great", context="domain1")
        temp_harness.remember("dogs are awesome", context="domain2")

        # Deactivate domain2
        temp_harness.deactivate_domain("domain2")

        # Search should only find cats
        results = temp_harness.recall("animals", k=10)

        # Should only get results from domain1
        assert all(r['domain_path'] == "domain1" for r in results)

    def test_list_domains(self, temp_harness):
        """Test listing all domains."""
        temp_harness.remember("test1", context="domain1")
        temp_harness.remember("test2", context="domain2")
        temp_harness.remember("test3", context="domain3")

        domains = temp_harness.list_domains()
        assert len(domains) == 3
        assert "domain1" in domains
        assert "domain2" in domains
        assert "domain3" in domains

    def test_get_domain_info(self, temp_harness):
        """Test getting domain information."""
        temp_harness.remember("test message", context="info/test")

        info = temp_harness.get_domain_info("info/test")

        assert info['domain_path'] == "info/test"
        assert info['message_count'] > 0
        assert 'domain_type' in info

    def test_clear_domain(self, temp_harness):
        """Test clearing/deleting a domain."""
        temp_harness.remember("test", context="to_delete")
        assert "to_delete" in temp_harness.manager.domains

        temp_harness.clear_domain("to_delete")
        assert "to_delete" not in temp_harness.manager.domains
        assert "to_delete" not in temp_harness.active_domains


class TestRecallModes:
    """Test different recall modes."""

    def test_semantic_mode(self, temp_harness):
        """Test semantic recall mode with provenance."""
        temp_harness.remember(
            "Geometric surfaces enable efficient memory",
            context="test/semantic"
        )

        results = temp_harness.recall(
            "memory efficiency",
            k=1,
            mode="semantic"
        )

        assert len(results) > 0
        # Semantic mode should include UV coordinates
        assert 'uv' in results[0]
        # And provenance
        assert 'provenance' in results[0]

    def test_simple_mode(self, temp_harness):
        """Test simple recall mode (fast, no provenance)."""
        temp_harness.remember(
            "Simple retrieval test",
            context="test/simple"
        )

        results = temp_harness.recall(
            "retrieval",
            k=1,
            mode="simple"
        )

        assert len(results) > 0
        # Simple mode doesn't include geometric details
        assert 'uv' not in results[0]
        assert 'provenance' not in results[0]

    def test_cross_domain_recall(self, temp_harness):
        """Test recall across multiple domains."""
        temp_harness.remember("Machine learning models", context="domain1")
        temp_harness.remember("Neural networks", context="domain2")
        temp_harness.remember("Deep learning architectures", context="domain3")

        results = temp_harness.recall("artificial intelligence", k=10)

        # Should get results from multiple domains
        domains_in_results = {r['domain_path'] for r in results}
        assert len(domains_in_results) > 1


class TestErrorHandling:
    """Test error handling."""

    def test_activate_nonexistent_domain(self, temp_harness):
        """Test activating a domain that doesn't exist."""
        with pytest.raises(ValueError):
            temp_harness.activate_domain("nonexistent/domain")

    def test_recall_with_no_active_domains(self, temp_harness):
        """Test recall with no active domains."""
        results = temp_harness.recall("test query", k=5)

        # Should return empty list, not error
        assert results == []

    def test_get_info_nonexistent_domain(self, temp_harness):
        """Test getting info for nonexistent domain."""
        with pytest.raises(Exception):  # DomainNotFoundError
            temp_harness.get_domain_info("nonexistent/domain")


class TestRepr:
    """Test string representations."""

    def test_repr(self, temp_harness):
        """Test harness __repr__."""
        temp_harness.remember("test", context="domain1")

        repr_str = repr(temp_harness)

        assert "MemoryHarness" in repr_str
        assert "1 domains" in repr_str or "domain" in repr_str
        assert "active" in repr_str
