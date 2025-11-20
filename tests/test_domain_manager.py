"""
Comprehensive tests for DomainManager.

Tests:
- Domain creation with hierarchy
- Domain discovery (list_domains, get_domain_info)
- Search (single domain, multi-domain, different modes)
- Mutation (add_to_domain)
- Lifecycle (load, unload, delete)
- Persistence (save/load)
- Error cases
"""

import pytest
from luna9.domain_manager import (
    DomainManager,
    DomainNotFoundError,
    DomainAlreadyExistsError,
    DomainInactiveError,
    InvalidDomainPathError
)
from luna9 import DomainType


def test_create_domains(dm):
    """Test domain creation with hierarchy."""
    # Create root level domain
    result = dm.create_domain("personal", "PERSONAL")
    assert result['status'] == 'created'
    assert result['path'] == 'personal'

    # Create hierarchical domains
    dm.create_domain("foundation", "FOUNDATION")
    dm.create_domain("foundation/books", "FOUNDATION")
    dm.create_domain("foundation/books/rust_book", "FOUNDATION")
    dm.create_domain("foundation/papers", "FOUNDATION")
    dm.create_domain("projects", "PROJECT")
    dm.create_domain("projects/luna_nine", "PROJECT")

    assert len(dm.domains) == 7


def test_list_domains(dm):
    """Test listing domains at different levels."""
    dm.create_domain("personal", "PERSONAL")
    dm.create_domain("foundation", "FOUNDATION")
    dm.create_domain("foundation/books", "FOUNDATION")
    dm.create_domain("foundation/books/rust_book", "FOUNDATION")
    dm.create_domain("foundation/papers", "FOUNDATION")
    dm.create_domain("projects", "PROJECT")

    # List root level
    root_domains = dm.list_domains()
    assert set(root_domains) == {"personal", "foundation", "projects"}

    # List children of foundation
    foundation_children = dm.list_domains("foundation")
    assert set(foundation_children) == {"foundation/books", "foundation/papers"}

    # List children of foundation/books
    books_children = dm.list_domains("foundation/books")
    assert books_children == ["foundation/books/rust_book"]


def test_get_domain_info(dm):
    """Test getting domain information."""
    dm.create_domain("personal", "PERSONAL")

    info = dm.get_domain_info("personal")

    assert info['name'] == 'personal'
    assert info['path'] == 'personal'
    assert info['type'] == 'personal'
    assert info['message_count'] == 0
    assert 'created_at' in info
    assert 'last_modified' in info


def test_add_to_domain(dm):
    """Test adding messages to domain."""
    dm.create_domain("test", "PROJECT")

    # Add messages
    messages = [
        "First message about testing",
        "Second message about domains",
        "Third message about memory"
    ]
    dm.add_to_domain("test", messages)

    info = dm.get_domain_info("test")
    assert info['message_count'] == 3

    # Add more messages
    dm.add_to_domain("test", ["Fourth message"])
    info = dm.get_domain_info("test")
    assert info['message_count'] == 4


def test_add_to_domain_with_metadata(dm):
    """Test adding messages with source attribution metadata."""
    dm.create_domain("test", "PROJECT")

    messages = ["Rust emphasizes memory safety"]
    metadata = [{
        "speaker": "agent",
        "source_type": "quoted",
        "source_attribution": {
            "type": "book",
            "title": "The Rust Programming Language",
            "author": "Steve Klabnik",
            "page": "4",
            "exact_quote": True
        }
    }]

    dm.add_to_domain("test", messages, metadata)

    info = dm.get_domain_info("test")
    assert info['message_count'] == 1


def test_search_single_domain(dm):
    """Test searching a single domain with different modes."""
    dm.create_domain("cooking", "PROJECT")

    messages = [
        "How to make pasta carbonara",
        "Pasta needs to be al dente",
        "Carbonara uses eggs and cheese",
        "Wine pairing for Italian food",
        "Best Italian restaurants"
    ]
    dm.add_to_domain("cooking", messages)

    # Semantic search
    results = dm.search_domain("cooking", "pasta dishes", k=3, mode="semantic")
    assert len(results) <= 3
    assert all('domain_path' in r for r in results)
    assert all('text' in r for r in results)
    assert all('score' in r for r in results)
    assert all('uv' in r for r in results)  # Semantic mode includes UV
    assert all('provenance' in r for r in results)

    # Literal search (should work similarly but without geometric data)
    results_literal = dm.search_domain("cooking", "pasta", k=3, mode="literal")
    assert len(results_literal) <= 3
    assert all('uv' not in r for r in results_literal)  # No UV in literal mode

    # Both mode
    results_both = dm.search_domain("cooking", "pasta", k=3, mode="both")
    assert all('uv' in r for r in results_both)  # Both mode includes UV


def test_search_multi_domain(dm):
    """Test searching across multiple domains."""
    # Create multiple domains with content
    dm.create_domain("foundation/books", "FOUNDATION")
    dm.create_domain("foundation/papers", "FOUNDATION")

    dm.add_to_domain("foundation/books", [
        "Rust ownership prevents memory errors",
        "Borrowing rules ensure safety",
        "Lifetimes track references"
    ])

    dm.add_to_domain("foundation/papers", [
        "Memory safety in systems programming",
        "Type systems prevent errors",
        "Linear types for resource management"
    ])

    # Search parent path (should search both children)
    results = dm.search_domain("foundation", "memory safety", k=5)

    # Should have results from both domains
    domain_paths = {r['domain_path'] for r in results}
    assert len(domain_paths) > 1  # Results from multiple domains

    # Results should be sorted by score
    scores = [r['score'] for r in results]
    assert scores == sorted(scores)


def test_persistence_save_load(dm):
    """Test domain persistence through save/load cycle."""
    # Create and populate domain
    dm.create_domain("foundation/books/rust", "FOUNDATION")

    messages = [
        "Rust emphasizes memory safety",
        "Ownership is Rust's key feature",
        "The borrow checker enforces safety"
    ]
    dm.add_to_domain("foundation/books/rust", messages)

    # Save to disk
    save_result = dm.save_domain("foundation/books/rust")
    assert save_result['status'] == 'saved'
    assert save_result['json_path']
    assert save_result['npz_path']

    # Query before unload
    results_before = dm.search_domain("foundation/books/rust", "memory", k=2)

    # Unload and create new manager instance
    dm.unload_domain("foundation/books/rust", save_first=False)
    dm2 = DomainManager(storage_dir=dm.storage_dir)

    # Load from disk
    load_result = dm2.load_domain("foundation/books/rust")
    assert load_result['status'] == 'loaded'

    # Query after load
    results_after = dm2.search_domain("foundation/books/rust", "memory", k=2)

    # Results should match
    assert len(results_before) == len(results_after)
    assert results_before[0]['text'] == results_after[0]['text']


def test_lifecycle_unload(dm):
    """Test unload with save."""
    dm.create_domain("test", "PROJECT")
    dm.add_to_domain("test", ["Some data"])

    # Unload with save
    result = dm.unload_domain("test", save_first=True)
    assert result['status'] == 'unloaded'
    assert result['saved'] is True

    # Domain should not appear in listings
    domains = dm.list_domains()
    assert "test" not in domains


def test_lifecycle_delete(dm):
    """Test deletion with confirmation."""
    dm.create_domain("temporary", "PROJECT")
    dm.add_to_domain("temporary", ["Some data"])

    # Try delete without confirmation (should fail)
    with pytest.raises(ValueError, match="confirm=True"):
        dm.delete_domain("temporary", confirm=False)

    # Delete with confirmation
    result = dm.delete_domain("temporary", confirm=True)
    assert result['status'] == 'deleted'
    assert result['message_count'] == 1

    # Domain should be gone
    assert "temporary" not in dm.domains

    # Try to access deleted domain (should fail)
    with pytest.raises(DomainNotFoundError):
        dm.get_domain_info("temporary")


def test_error_domain_not_found(dm):
    """Test DomainNotFoundError cases."""
    # get_domain_info on missing domain
    with pytest.raises(DomainNotFoundError, match="nonexistent"):
        dm.get_domain_info("nonexistent")

    # search_domain on missing domain
    with pytest.raises(DomainNotFoundError):
        dm.search_domain("nonexistent", "query")

    # add_to_domain on missing domain
    with pytest.raises(DomainNotFoundError):
        dm.add_to_domain("nonexistent", ["message"])


def test_error_domain_already_exists(dm):
    """Test DomainAlreadyExistsError."""
    dm.create_domain("test", "PROJECT")

    # Try to create same domain again
    with pytest.raises(DomainAlreadyExistsError, match="already exists"):
        dm.create_domain("test", "PROJECT")


def test_error_invalid_path(dm):
    """Test InvalidDomainPathError."""
    # Empty path
    with pytest.raises(InvalidDomainPathError):
        dm.create_domain("", "PROJECT")

    # Path too deep
    with pytest.raises(InvalidDomainPathError, match="too deep"):
        dm.create_domain("a/b/c/d", "PROJECT")

    # Invalid characters
    with pytest.raises(InvalidDomainPathError):
        dm.create_domain("test domain", "PROJECT")  # Space


def test_error_domain_inactive(dm):
    """Test DomainInactiveError."""
    dm.create_domain("test", "PROJECT")
    dm.add_to_domain("test", ["Some data"])

    # Unload domain
    dm.unload_domain("test", save_first=False)

    # Try to search inactive domain
    with pytest.raises(DomainInactiveError, match="inactive"):
        dm.search_domain("test", "query")


def test_real_world_scenario(dm):
    """Test realistic usage scenario."""
    # Set up knowledge base
    dm.create_domain("personal", "PERSONAL")
    dm.create_domain("foundation", "FOUNDATION")
    dm.create_domain("foundation/books", "FOUNDATION")
    dm.create_domain("foundation/books/rust", "FOUNDATION")
    dm.create_domain("projects", "PROJECT")
    dm.create_domain("projects/luna_nine", "PROJECT")

    # Add content
    dm.add_to_domain("personal", [
        "Working on Luna Nine project",
        "Building geometric memory system",
        "Focus on continuous context"
    ])

    dm.add_to_domain("foundation/books/rust", [
        "Ownership is Rust's key feature",
        "Borrowing allows safe references",
        "Lifetimes prevent dangling pointers"
    ])

    dm.add_to_domain("projects/luna_nine", [
        "Luna Nine uses geometric memory",
        "Bézier surfaces represent semantic space",
        "Path curvature measures relationships"
    ])

    # Query across foundation
    results = dm.search_domain("foundation", "Rust memory", k=3)
    assert len(results) > 0
    assert any("rust" in r['domain_path'].lower() for r in results)

    # Query personal
    results = dm.search_domain("personal", "Luna Nine", k=2)
    assert len(results) > 0
    assert all(r['domain_path'] == 'personal' for r in results)

    # List all domains
    all_domains = []
    for root in dm.list_domains():
        all_domains.append(root)
        for child in dm.list_domains(root):
            all_domains.append(child)

    assert len(all_domains) >= 6
