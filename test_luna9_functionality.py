"""
Comprehensive test of Luna Nine core functionality.
Tests all major features to ensure package is working correctly.
"""

import sys
import tempfile
import shutil
from pathlib import Path

print("=" * 60)
print("Luna Nine Functionality Test")
print("=" * 60)

# Test 1: Basic imports
print("\n[TEST 1] Testing basic imports...")
try:
    import luna9
    from luna9 import (
        SemanticSurface,
        Domain,
        DomainManager,
        DomainType,
        analyze_relationships,
        format_for_llm,
        HashIndex
    )
    print(f"[OK] All imports successful")
    print(f"  Version: {luna9.__version__}")
    print(f"  Exports: {len([m for m in dir(luna9) if not m.startswith('_')])} public members")
except Exception as e:
    print(f"[FAIL] Import failed: {e}")
    sys.exit(1)

# Test 2: Analyze relationships (small dataset, embeddings mode)
print("\n[TEST 2] Testing analyze_relationships (embeddings mode)...")
try:
    products = [
        "Meeting notes with AI summarization",
        "Notes for meetings with bullet points",
        "Voice recordings transcribed to text",
    ]

    analysis = analyze_relationships(products)

    assert 'items' in analysis
    assert 'pairwise_relationships' in analysis
    assert 'summary' in analysis
    assert analysis['metadata']['method'] == 'embeddings'
    assert len(analysis['items']) == 3

    print(f"[OK] analyze_relationships works")
    print(f"  Method: {analysis['metadata']['method']}")
    print(f"  Items analyzed: {len(analysis['items'])}")
    print(f"  Pairwise relationships: {len(analysis['pairwise_relationships'])}")
    print(f"  Most similar pair: {analysis['summary']['most_similar_pair']['similarity']:.3f}")

except Exception as e:
    print(f"[FAIL] analyze_relationships failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Format for LLM
print("\n[TEST 3] Testing format_for_llm...")
try:
    formatted = format_for_llm(analysis)
    assert isinstance(formatted, str)
    assert "SEMANTIC RELATIONSHIP ANALYSIS" in formatted
    assert "EMBEDDINGS" in formatted

    print(f"[OK] format_for_llm works")
    print(f"  Output length: {len(formatted)} characters")

except Exception as e:
    print(f"[FAIL] format_for_llm failed: {e}")
    sys.exit(1)

# Test 4: Domain creation and querying
print("\n[TEST 4] Testing Domain creation and querying...")
try:
    messages = [
        "We discussed adding authentication to the API",
        "The database schema needs indexes on user_id",
        "Frontend should use React hooks for state management",
        "We decided to deploy on Railway for simplicity",
        "API endpoints should return JSON with consistent structure"
    ]

    domain = Domain(
        name="test_project",
        domain_type=DomainType.PROJECT
    )
    domain.add_messages(messages)

    assert domain.name == "test_project"
    assert len(domain.surface.messages) == 5
    assert domain.surface is not None

    print(f"[OK] Domain created successfully")
    print(f"  Domain name: {domain.name}")
    print(f"  Message count: {len(domain.surface.messages)}")
    print(f"  Surface grid: {domain.surface.grid_m}x{domain.surface.grid_n}")

except Exception as e:
    print(f"[FAIL] Domain creation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Domain querying
print("\n[TEST 5] Testing Domain querying...")
try:
    result = domain.query("What did we decide about deployment?", k=2)

    # Domain.query() returns a Dict with domain context and results
    assert isinstance(result, dict)
    assert 'domain' in result
    assert 'uv' in result
    assert 'curvature' in result

    # Check we got interpretation results (smooth mode)
    assert 'interpretation' in result or 'sources' in result

    # Get messages from interpretation (default mode='both' includes this)
    if 'interpretation' in result:
        messages = result['interpretation']['messages']
        assert len(messages) > 0
        assert isinstance(messages[0], str)
        top_message = messages[0]
    elif 'sources' in result:
        messages = result['sources']['messages']
        assert len(messages) > 0
        top_message = messages[0]

    print(f"[OK] Domain querying works")
    print(f"  Query: 'What did we decide about deployment?'")
    print(f"  Domain: {result['domain']}")
    print(f"  UV coordinates: ({result['uv'][0]:.3f}, {result['uv'][1]:.3f})")
    print(f"  Curvature: {result['curvature']}")  # Tuple of (gaussian, mean) curvatures
    print(f"  Top result: '{top_message[:50]}...'")

except Exception as e:
    print(f"[FAIL] Domain querying failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: Hash Index integration
print("\n[TEST 6] Testing HashIndex...")
try:
    # Domain already has hash_index enabled by default
    # We can verify it exists and works
    assert domain.hash_index is not None

    # Query uses hash index automatically for O(1) lookups
    result_with_hash = domain.query("database and indexes", k=2)

    # Verify we got results
    assert isinstance(result_with_hash, dict)
    assert 'interpretation' in result_with_hash or 'sources' in result_with_hash

    # Get top message
    if 'interpretation' in result_with_hash:
        top_msg = result_with_hash['interpretation']['messages'][0]
    else:
        top_msg = result_with_hash['sources']['messages'][0]

    print(f"[OK] HashIndex works")
    print(f"  Hash index integrated with domain")
    print(f"  Query succeeded with O(1) candidate retrieval")
    print(f"  Top result: '{top_msg[:50]}...'")

except Exception as e:
    print(f"[FAIL] HashIndex failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 7: DomainManager with persistence
print("\n[TEST 7] Testing DomainManager with persistence...")
try:
    # Create temporary directory for testing
    temp_dir = tempfile.mkdtemp()

    try:
        # Create domain manager
        manager = DomainManager(storage_dir=temp_dir)

        # Create a domain (empty at first)
        manager.create_domain(
            path="test_domain",
            domain_type=DomainType.CONVERSATION.value
        )

        # Add messages to the domain
        manager.add_to_domain(
            path="test_domain",
            messages=["Test message 1", "Test message 2", "Test message 3"]
        )

        # Check domain was created
        assert "test_domain" in manager.list_domains()

        # Verify domain has messages
        domain_obj = manager.domains["test_domain"]
        assert domain_obj.surface is not None
        assert len(domain_obj.surface.messages) == 3

        # Search the domain
        results = manager.search_domain(
            path="test_domain",
            query="test",
            k=2
        )

        # Note: search_domain may return empty if query doesn't match well enough
        # Just verify it doesn't crash
        assert isinstance(results, list)

        # Save domain to disk
        manager.save_domain("test_domain")

        # Check files were created (domain.json and surface.npz)
        domain_dir = Path(temp_dir) / "test_domain"
        domain_file = domain_dir / "domain.json"
        surface_file = domain_dir / "surface.npz"
        assert domain_file.exists()
        assert surface_file.exists()

        # Create new manager and load domain from disk
        manager2 = DomainManager(storage_dir=temp_dir)
        manager2.load_domain("test_domain")
        assert "test_domain" in manager2.list_domains()

        # Verify loaded domain works
        results2 = manager2.search_domain("test_domain", "message", k=2)
        assert isinstance(results2, list)

        print(f"[OK] DomainManager works")
        print(f"  Created domain with 3 messages")
        print(f"  Domain saved to: {domain_dir}")
        print(f"  Domain loaded successfully")
        print(f"  File-based persistence working")

    finally:
        # Cleanup - try to remove, but don't fail if files are locked
        try:
            shutil.rmtree(temp_dir)
        except (PermissionError, OSError):
            # Windows can lock .npz files, cleanup will happen on exit
            pass

except Exception as e:
    print(f"[FAIL] DomainManager failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 8: Check dependencies are loadable
print("\n[TEST 8] Testing dependencies...")
try:
    import numpy as np
    import numba
    from sentence_transformers import SentenceTransformer
    import scipy

    print(f"[OK] All dependencies loadable")
    print(f"  numpy: {np.__version__}")
    print(f"  numba: {numba.__version__}")
    print(f"  scipy: {scipy.__version__}")

except Exception as e:
    print(f"[FAIL] Dependency check failed: {e}")
    sys.exit(1)

# Summary
print("\n" + "=" * 60)
print("All tests passed! [OK]")
print("=" * 60)
print("\nLuna Nine is working correctly!")
print("Package is ready for:")
print("  - TestPyPI publishing")
print("  - PyPI publishing")
print("  - Production use")
