"""
Comprehensive tests for DomainManager.

Tests:
- Domain creation with hierarchy
- Domain discovery (list_domains, get_domain_info)
- Search (single domain, multi-domain, different modes)
- Mutation (add_to_domain)
- Lifecycle (load, unload, delete)
- Error cases
"""

import sys
from pathlib import Path

# Add luna9-python to path
sys.path.insert(0, str(Path(__file__).parent / "luna9-python"))

from luna9.domain_manager import (
    DomainManager,
    DomainNotFoundError,
    DomainAlreadyExistsError,
    DomainInactiveError,
    InvalidDomainPathError
)
from luna9 import DomainType


def test_create_domains():
    """Test domain creation with hierarchy."""
    print("Testing domain creation...")

    dm = DomainManager()

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

    print("✓ Domain creation works")


def test_list_domains():
    """Test listing domains at different levels."""
    print("Testing list_domains...")

    dm = DomainManager()
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

    print("✓ list_domains works at all levels")


def test_get_domain_info():
    """Test getting domain information."""
    print("Testing get_domain_info...")

    dm = DomainManager()
    dm.create_domain("personal", "PERSONAL")

    info = dm.get_domain_info("personal")

    assert info['name'] == 'personal'
    assert info['path'] == 'personal'
    assert info['type'] == 'personal'
    assert info['message_count'] == 0
    assert 'created_at' in info
    assert 'last_modified' in info

    print("✓ get_domain_info works")


def test_add_to_domain():
    """Test adding messages to domain."""
    print("Testing add_to_domain...")

    dm = DomainManager()
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

    print("✓ add_to_domain works")


def test_search_single_domain():
    """Test searching a single domain with different modes."""
    print("Testing single domain search...")

    dm = DomainManager()
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

    print("✓ Single domain search works with all modes")


def test_search_multi_domain():
    """Test searching across multiple domains."""
    print("Testing multi-domain search...")

    dm = DomainManager()

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

    print("✓ Multi-domain search works")


def test_lifecycle_load_unload():
    """Test load/unload lifecycle."""
    print("Testing load/unload...")

    dm = DomainManager()
    dm.create_domain("test", "PROJECT")

    # Load domain (currently no-op)
    status = dm.load_domain("test")
    assert status['status'] in ('loaded', 'already_loaded')

    # Unload domain
    dm.unload_domain("test")

    # Domain should not appear in listings
    domains = dm.list_domains()
    assert "test" not in domains

    # But should still exist (can be reloaded)
    assert "test" in dm.domains

    # Reload
    dm.load_domain("test")
    domains = dm.list_domains()
    assert "test" in domains

    print("✓ Load/unload lifecycle works")


def test_lifecycle_delete():
    """Test deletion with confirmation."""
    print("Testing delete...")

    dm = DomainManager()
    dm.create_domain("temporary", "PROJECT")
    dm.add_to_domain("temporary", ["Some data"])

    # Try delete without confirmation (should fail)
    try:
        dm.delete_domain("temporary", confirm=False)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "confirm=True" in str(e)

    # Delete with confirmation
    result = dm.delete_domain("temporary", confirm=True)
    assert result['status'] == 'deleted'
    assert result['message_count'] == 1

    # Domain should be gone
    assert "temporary" not in dm.domains

    # Try to access deleted domain (should fail)
    try:
        dm.get_domain_info("temporary")
        assert False, "Should have raised DomainNotFoundError"
    except DomainNotFoundError:
        pass

    print("✓ Delete with confirmation works")


def test_error_domain_not_found():
    """Test DomainNotFoundError cases."""
    print("Testing DomainNotFoundError...")

    dm = DomainManager()

    # get_domain_info on missing domain
    try:
        dm.get_domain_info("nonexistent")
        assert False, "Should have raised DomainNotFoundError"
    except DomainNotFoundError as e:
        assert "nonexistent" in str(e)

    # search_domain on missing domain
    try:
        dm.search_domain("nonexistent", "query")
        assert False, "Should have raised DomainNotFoundError"
    except DomainNotFoundError:
        pass

    # add_to_domain on missing domain
    try:
        dm.add_to_domain("nonexistent", ["message"])
        assert False, "Should have raised DomainNotFoundError"
    except DomainNotFoundError:
        pass

    print("✓ DomainNotFoundError raised correctly")


def test_error_domain_already_exists():
    """Test DomainAlreadyExistsError."""
    print("Testing DomainAlreadyExistsError...")

    dm = DomainManager()
    dm.create_domain("test", "PROJECT")

    # Try to create same domain again
    try:
        dm.create_domain("test", "PROJECT")
        assert False, "Should have raised DomainAlreadyExistsError"
    except DomainAlreadyExistsError as e:
        assert "already exists" in str(e)

    print("✓ DomainAlreadyExistsError raised correctly")


def test_error_invalid_path():
    """Test InvalidDomainPathError."""
    print("Testing InvalidDomainPathError...")

    dm = DomainManager()

    # Empty path
    try:
        dm.create_domain("", "PROJECT")
        assert False, "Should have raised InvalidDomainPathError"
    except InvalidDomainPathError:
        pass

    # Path too deep
    try:
        dm.create_domain("a/b/c/d", "PROJECT")
        assert False, "Should have raised InvalidDomainPathError"
    except InvalidDomainPathError as e:
        assert "too deep" in str(e)

    # Invalid characters
    try:
        dm.create_domain("test domain", "PROJECT")  # Space
        assert False, "Should have raised InvalidDomainPathError"
    except InvalidDomainPathError:
        pass

    print("✓ InvalidDomainPathError raised correctly")


def test_error_domain_inactive():
    """Test DomainInactiveError."""
    print("Testing DomainInactiveError...")

    dm = DomainManager()
    dm.create_domain("test", "PROJECT")
    dm.add_to_domain("test", ["Some data"])

    # Unload domain
    dm.unload_domain("test")

    # Try to search inactive domain
    try:
        dm.search_domain("test", "query")
        assert False, "Should have raised DomainInactiveError"
    except DomainInactiveError as e:
        assert "inactive" in str(e).lower()

    print("✓ DomainInactiveError raised correctly")


def test_real_world_scenario():
    """Test realistic usage scenario."""
    print("Testing real-world scenario...")

    dm = DomainManager()

    # Set up knowledge base
    dm.create_domain("personal", "PERSONAL")
    dm.create_domain("foundation", "FOUNDATION")
    dm.create_domain("foundation/books", "FOUNDATION")
    dm.create_domain("foundation/books/rust", "FOUNDATION")
    dm.create_domain("projects", "PROJECT")
    dm.create_domain("projects/luna_nine", "PROJECT")

    # Add content
    dm.add_to_domain("personal", [
        "Alicia is my collaborator",
        "We're building Luna Nine together",
        "She values presence over productivity"
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
    results = dm.search_domain("personal", "Alicia", k=2)
    assert len(results) > 0
    assert all(r['domain_path'] == 'personal' for r in results)

    # List all domains
    all_domains = []
    for root in dm.list_domains():
        all_domains.append(root)
        for child in dm.list_domains(root):
            all_domains.append(child)

    assert len(all_domains) >= 6

    print("✓ Real-world scenario works")


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("DomainManager Test Suite")
    print("=" * 60)

    tests = [
        test_create_domains,
        test_list_domains,
        test_get_domain_info,
        test_add_to_domain,
        test_search_single_domain,
        test_search_multi_domain,
        test_lifecycle_load_unload,
        test_lifecycle_delete,
        test_error_domain_not_found,
        test_error_domain_already_exists,
        test_error_invalid_path,
        test_error_domain_inactive,
        test_real_world_scenario
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"✗ {test.__name__} FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
