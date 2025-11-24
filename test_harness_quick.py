#!/usr/bin/env python3
"""Quick manual test of MemoryHarness."""

import tempfile
import shutil
import sys
from luna9 import MemoryHarness

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# Create temp directory
temp_dir = tempfile.mkdtemp()
print(f"Testing in: {temp_dir}\n")

try:
    # Create harness
    print("1. Creating MemoryHarness...")
    harness = MemoryHarness(base_path=temp_dir)
    print(f"   ✓ Created: {harness}\n")

    # Remember something (need at least 2 messages per domain for search to work)
    print("2. Remembering information...")
    harness.remember(
        "Luna 9 uses geometric surfaces for semantic memory",
        context="test/concepts"
    )
    harness.remember(
        "Bézier surfaces enable efficient queries",
        context="test/concepts"
    )
    harness.remember(
        "The system has sub-linear query scaling",
        context="test/performance"
    )
    harness.remember(
        "Hash indexing provides O(1) candidate retrieval",
        context="test/performance"
    )
    print(f"   ✓ Added 4 messages across 2 domains")
    print(f"   ✓ Active domains: {harness.active_domain_count}\n")

    # Check domains exist
    print("3. Checking domains...")
    domains = list(harness.manager.domains.keys())
    print(f"   Domains in memory: {domains}")
    for domain_path in domains:
        domain = harness.manager.domains[domain_path]
        surface_msgs = len(domain.surface.messages) if domain.surface else 0
        print(f"   - {domain_path}: active={domain._active}, surface={domain.surface is not None}, surface_msgs={surface_msgs}")
    print()

    # Try recall
    print("4. Testing recall...")
    print(f"   Active domains: {harness.active_domains}")

    # Try searching directly with manager
    print("   Direct search test...")
    for domain_path in harness.active_domains:
        direct_results = harness.manager.search_domain(domain_path, "geometric", k=3, mode="semantic")
        print(f"     {domain_path}: {len(direct_results)} results")

    results = harness.recall("geometric memory", k=3)
    print(f"   Recall found {len(results)} results:")
    for i, result in enumerate(results):
        print(f"   {i+1}. [{result['domain_path']}] Score: {result['score']:.3f}")
        print(f"      Text: {result['text'][:60]}...")
    print()

    # Ingest document
    print("5. Testing document ingestion...")
    test_doc = """
Chapter 1: Introduction

This is a test document with multiple sections.
It demonstrates the chunking capability.

Chapter 2: Features

Luna 9 has several key features.
Geometric memory is the core innovation.

Chapter 3: Conclusion

This concludes our test document.
    """.strip()

    chunk_count = harness.ingest_document(
        test_doc,
        domain="test/document",
        title="Test Document"
    )
    print(f"   ✓ Ingested {chunk_count} chunks\n")

    # Query the document
    print("6. Querying ingested document...")
    results = harness.recall("features and innovation", k=2)
    print(f"   Found {len(results)} results:")
    for i, result in enumerate(results):
        print(f"   {i+1}. [{result['domain_path']}] Score: {result['score']:.3f}")
        print(f"      Text: {result['text'][:60]}...")
    print()

    print("✓ All manual tests passed!")

except Exception as e:
    print(f"\n✗ Error: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

finally:
    # Cleanup
    shutil.rmtree(temp_dir)
    print(f"\nCleaned up: {temp_dir}")
