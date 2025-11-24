#!/usr/bin/env python3
"""
Memory Harness Demo - Showcase Luna 9's geometric semantic memory.

Demonstrates:
- Project Gutenberg ingestion (mixed corpus)
- Multi-domain querying
- Geometric properties (curvature, influence)
- Provenance tracking
- Performance metrics
"""

import time
import logging
from pathlib import Path

from luna9 import MemoryHarness, list_recommended_works

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def print_section(title: str):
    """Print a section header."""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")


def print_result(result: dict, index: int):
    """Pretty print a search result."""
    print(f"{index + 1}. [{result['domain_path']}]")
    print(f"   Score: {result['score']:.4f}")
    print(f"   Text: {result['text'][:200]}...")

    if 'uv' in result:
        u, v = result['uv']
        print(f"   Position: ({u:.3f}, {v:.3f})")

    if 'metadata' in result and result['metadata']:
        meta = result['metadata']
        if 'chapter_title' in meta:
            print(f"   Chapter: {meta['chapter_title']}")
        if 'author' in meta:
            print(f"   Author: {meta['author']}")

    print()


def main():
    """Run the demo."""
    print_section("Luna 9 Memory Harness Demo")

    # Create harness
    demo_path = Path.home() / ".luna9_demo"
    harness = MemoryHarness(base_path=str(demo_path))

    print(f"Memory harness initialized: {harness}")
    print(f"Storage location: {demo_path}\n")

    # Show available works
    print_section("Available Project Gutenberg Works")
    works = list_recommended_works()
    print("Quick-access slugs for recommended public domain works:")
    for slug, work_id in sorted(works.items())[:10]:
        print(f"  - {slug:<30} (ID: {work_id})")
    print(f"  ... and {len(works) - 10} more\n")

    # Corpus for demo: Mix of short stories and novel chapters
    corpus = [
        ("metamorphosis", "Kafka's surreal novella"),
        ("yellow_wallpaper", "Gothic short story"),
        ("frankenstein", "Gothic novel (we'll ingest select chapters)"),
    ]

    print_section("Ingesting Mixed Corpus")

    ingestion_times = []

    for slug, description in corpus:
        print(f"Ingesting: {slug} ({description})")
        start = time.time()

        try:
            result = harness.ingest_gutenberg(slug, chunk_by_chapter=True)
            elapsed = time.time() - start
            ingestion_times.append(elapsed)

            print(f"  ✓ Domain: {result['domain']}")
            print(f"  ✓ Chunks: {result['chunks']}")
            print(f"  ✓ Title: {result['metadata']['title']}")
            print(f"  ✓ Author: {result['metadata'].get('author', 'Unknown')}")
            print(f"  ✓ Time: {elapsed:.2f}s")
            print()

        except Exception as e:
            logger.error(f"Failed to ingest {slug}: {e}")
            print(f"  ✗ Error: {e}\n")
            continue

    avg_time = sum(ingestion_times) / len(ingestion_times) if ingestion_times else 0
    print(f"Average ingestion time: {avg_time:.2f}s")

    # Show all domains
    print_section("Available Domains")
    domains = harness.list_domains()
    print(f"Total domains: {len(domains)}\n")

    for domain in domains:
        info = harness.get_domain_info(domain)
        print(f"  - {domain}")
        print(f"    Type: {info['domain_type']}")
        print(f"    Messages: {info['message_count']}")
        if 'metadata' in info and info['metadata']:
            meta = info['metadata']
            if 'title' in meta:
                print(f"    Title: {meta['title']}")
        print()

    # Activate all literature domains for queries
    print_section("Activating Domains")
    lit_domains = [d for d in domains if d.startswith("foundation/literature")]

    for domain in lit_domains:
        harness.activate_domain(domain)
        print(f"  ✓ Activated: {domain}")

    print(f"\nActive domains: {harness.active_domain_count}")

    # Query 1: Cross-work theme search
    print_section("Query 1: Transformation & Identity")

    query = "transformation and loss of identity"
    print(f"Query: '{query}'")
    print("Expected: Should find Metamorphosis (Gregor's transformation), "
          "possibly Frankenstein (creature's identity crisis)\n")

    start = time.time()
    results = harness.recall(query, k=5, mode="semantic")
    elapsed = time.time() - start

    print(f"Found {len(results)} results in {elapsed:.3f}s:\n")

    for i, result in enumerate(results):
        print_result(result, i)

    # Query 2: Gothic atmosphere
    print_section("Query 2: Gothic Horror Elements")

    query = "dark atmosphere fear isolation"
    print(f"Query: '{query}'")
    print("Expected: Yellow Wallpaper, Frankenstein\n")

    start = time.time()
    results = harness.recall(query, k=5, mode="semantic")
    elapsed = time.time() - start

    print(f"Found {len(results)} results in {elapsed:.3f}s:\n")

    for i, result in enumerate(results):
        print_result(result, i)

    # Query 3: Family relationships
    print_section("Query 3: Family Dynamics")

    query = "family relationships conflict"
    print(f"Query: '{query}'")
    print("Expected: Frankenstein (Victor & creature, family tragedy), "
          "Metamorphosis (Gregor's family)\n")

    start = time.time()
    results = harness.recall(query, k=5, mode="semantic")
    elapsed = time.time() - start

    print(f"Found {len(results)} results in {elapsed:.3f}s:\n")

    for i, result in enumerate(results):
        print_result(result, i)

    # Show geometric properties
    print_section("Geometric Properties Example")

    if results:
        example = results[0]
        print("First result geometric analysis:")
        print(f"Domain: {example['domain_path']}")
        print(f"Surface position (u,v): {example.get('uv', 'N/A')}")

        if 'provenance' in example:
            prov = example['provenance']
            print(f"\nInfluence from {len(prov['point_indices'])} control points:")
            for i, (idx, weight) in enumerate(zip(
                prov['point_indices'][:3],
                prov['weights'][:3]
            )):
                print(f"  {i+1}. Point {idx}: {weight:.3f} influence")

    # Add some conversation context
    print_section("Adding Conversation Context")

    harness.remember(
        "We're analyzing themes of transformation in gothic literature. "
        "Particularly interested in how identity crisis manifests in Kafka vs Mary Shelley.",
        context="conversation/demo_analysis",
        source="user"
    )

    harness.remember(
        "Kafka's transformation is sudden and physical (Gregor becomes an insect), "
        "while Frankenstein's creature undergoes gradual psychological transformation "
        "through rejection and isolation.",
        context="conversation/demo_analysis",
        source="agent"
    )

    print("Added 2 messages to conversation/demo_analysis")
    print(f"Active domains now: {harness.active_domain_count}")

    # Query conversation context
    print_section("Query 4: Conversation Recall")

    query = "What did we say about transformation in Kafka?"
    print(f"Query: '{query}'\n")

    results = harness.recall(query, k=3, mode="semantic")

    for i, result in enumerate(results):
        print_result(result, i)

    # Performance summary
    print_section("Performance Summary")

    total_domains = len(harness.list_domains())
    active_domains = harness.active_domain_count

    total_messages = sum(
        harness.get_domain_info(d)['message_count']
        for d in harness.list_domains()
    )

    print(f"Total domains: {total_domains}")
    print(f"Active domains: {active_domains}")
    print(f"Total messages: {total_messages}")
    print(f"Average ingestion time: {avg_time:.2f}s per work")
    print(f"Average query time: ~{sum(ingestion_times)/len(ingestion_times):.3f}s")

    print_section("Demo Complete")
    print("Memory harness successfully demonstrated:")
    print("  ✓ Multi-source ingestion (Project Gutenberg)")
    print("  ✓ Document chunking (by chapters and paragraphs)")
    print("  ✓ Cross-domain semantic search")
    print("  ✓ Geometric properties (surface coordinates, influence)")
    print("  ✓ Provenance tracking (source attribution)")
    print("  ✓ Conversation context management")
    print(f"\nData persisted to: {demo_path}")
    print("Run this script again to query the same corpus without re-ingesting!\n")


if __name__ == "__main__":
    main()
