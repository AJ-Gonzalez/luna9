"""
Tests for Domain class functionality.

Tests:
- Creating empty domains
- Adding messages incrementally
- Creating domains from initial messages
- Querying domains
- Domain metadata
"""

import pytest
from luna9 import Domain, DomainType


def test_create_empty_domain():
    """Test creating an empty domain."""
    domain = Domain.create_empty("personal", DomainType.PERSONAL)

    assert domain.name == "personal"
    assert domain.path == "personal"
    assert domain.domain_type == DomainType.PERSONAL
    assert domain.surface is None

    info = domain.get_info()
    assert info['message_count'] == 0


def test_add_message_to_empty_domain():
    """Test adding first message to empty domain."""
    domain = Domain.create_empty("test", DomainType.PROJECT)

    # Add first message
    domain.add_message("First message")

    assert domain.surface is not None
    assert len(domain.surface.messages) == 1
    assert domain.surface.messages[0] == "First message"

    info = domain.get_info()
    assert info['message_count'] == 1


def test_add_messages_incrementally():
    """Test adding messages one by one to domain."""
    domain = Domain.create_empty("personal", DomainType.PERSONAL)

    # Add messages incrementally
    domain.add_message("I love building with you")
    domain.add_message("We're building Luna Nine together")
    domain.add_message("Your praise makes me melt")

    assert len(domain.surface.messages) == 3
    info = domain.get_info()
    assert info['message_count'] == 3


def test_add_messages_batch():
    """Test adding multiple messages at once."""
    domain = Domain.create_empty("test", DomainType.PROJECT)

    messages = [
        "Message one",
        "Message two",
        "Message three",
        "Message four"
    ]
    domain.add_messages(messages)

    assert len(domain.surface.messages) == 4
    assert domain.surface.messages == messages


def test_add_message_with_metadata():
    """Test adding messages with metadata."""
    domain = Domain.create_empty("test", DomainType.PROJECT)

    metadata = {
        "speaker": "user",
        "source_type": "original",
        "timestamp": "2025-11-20"
    }

    domain.add_message("Test message", metadata=metadata)

    assert len(domain._message_metadata) == 1
    assert domain._message_metadata[0] == metadata


def test_create_from_messages():
    """Test creating domain with initial messages."""
    messages = [
        "Rust has ownership rules",
        "Borrowing allows references",
        "Lifetimes ensure validity",
        "The borrow checker prevents data races"
    ]

    domain = Domain.create_from_messages(
        name="rust_book",
        domain_type=DomainType.FOUNDATION,
        messages=messages,
        parent_path="foundation"
    )

    assert domain.name == "rust_book"
    assert domain.path == "foundation/rust_book"
    assert len(domain.surface.messages) == 4
    assert domain.surface.messages == messages


def test_create_from_messages_with_metadata():
    """Test creating domain with messages and metadata."""
    messages = ["Message 1", "Message 2"]
    metadata = [
        {"speaker": "user"},
        {"speaker": "agent"}
    ]

    domain = Domain.create_from_messages(
        name="test",
        domain_type=DomainType.PROJECT,
        messages=messages,
        message_metadata=metadata
    )

    assert len(domain._message_metadata) == 2
    assert domain._message_metadata == metadata


def test_query_domain():
    """Test querying domain for relevant messages."""
    messages = [
        "Rust ownership prevents memory errors",
        "Python has garbage collection",
        "JavaScript is dynamically typed",
        "Memory safety is important",
        "Type systems catch bugs early"
    ]

    domain = Domain.create_from_messages(
        name="programming",
        domain_type=DomainType.FOUNDATION,
        messages=messages
    )

    # Query for memory-related content
    results = domain.query("memory management", k=3)

    assert 'domain' in results
    assert results['domain'] == 'programming'
    assert 'uv' in results
    assert 'curvature' in results

    # Should have both interpretation and sources (mode='both' is default)
    assert 'interpretation' in results or 'sources' in results


def test_query_empty_domain():
    """Test querying empty domain returns appropriate response."""
    domain = Domain.create_empty("empty", DomainType.PROJECT)

    results = domain.query("anything", k=5)

    assert 'domain' in results
    assert 'message' in results
    assert results['message'] == 'Domain is empty'


def test_domain_hierarchy():
    """Test hierarchical domain paths."""
    # Root level
    root = Domain.create_empty("personal", DomainType.PERSONAL)
    assert root.path == "personal"
    assert root.parent_path is None

    # Second level
    child = Domain.create_empty("books", DomainType.FOUNDATION, parent_path="foundation")
    assert child.path == "foundation/books"
    assert child.parent_path == "foundation"

    # Third level
    grandchild = Domain.create_empty(
        "rust",
        DomainType.FOUNDATION,
        parent_path="foundation/books"
    )
    assert grandchild.path == "foundation/books/rust"
    assert grandchild.parent_path == "foundation/books"


def test_domain_metadata():
    """Test domain metadata storage."""
    custom_metadata = {
        "author": "Alicia",
        "project": "Luna Nine",
        "purpose": "Memory testing"
    }

    domain = Domain.create_empty(
        "test",
        DomainType.PROJECT,
        metadata=custom_metadata
    )

    assert domain.metadata == custom_metadata

    info = domain.get_info()
    assert info['metadata'] == custom_metadata


def test_domain_timestamps():
    """Test domain timestamp tracking."""
    domain = Domain.create_empty("test", DomainType.PROJECT)

    created_at = domain.created_at
    last_modified = domain.last_modified

    # Initially should be equal
    assert created_at == last_modified

    # Add message and check last_modified updates
    import time
    time.sleep(0.01)  # Small delay to ensure time difference
    domain.add_message("Update message")

    assert domain.last_modified > created_at


def test_domain_info_complete():
    """Test get_info returns complete domain information."""
    messages = ["Test message 1", "Test message 2", "Test message 3"]
    domain = Domain.create_from_messages(
        name="test",
        domain_type=DomainType.PROJECT,
        messages=messages
    )

    info = domain.get_info()

    # Check all expected fields present
    assert 'name' in info
    assert 'path' in info
    assert 'type' in info
    assert 'created_at' in info
    assert 'last_modified' in info
    assert 'message_count' in info
    assert 'grid_size' in info
    assert 'metadata' in info

    # Check values
    assert info['name'] == 'test'
    assert info['path'] == 'test'
    assert info['type'] == 'project'
    assert info['message_count'] == 3
    assert info['grid_size'] == '1x3'  # Grid for 3 messages


def test_domain_repr():
    """Test domain string representation."""
    domain = Domain.create_from_messages(
        name="test",
        domain_type=DomainType.PROJECT,
        messages=["Message 1", "Message 2"]
    )

    repr_str = repr(domain)

    assert "Domain" in repr_str
    assert "test" in repr_str
    assert "project" in repr_str
    assert "messages=2" in repr_str
