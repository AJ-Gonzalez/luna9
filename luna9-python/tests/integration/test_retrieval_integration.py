#!/usr/bin/env python3
"""
Quick test to verify that full retrieved content appears in initiative context.

This tests that our wireframe+texture integration works correctly.
"""

from luna9 import Domain, DomainType, InitiativeEngine, BoundariesConfig

# Create test domain with sample content
domain = Domain.create_empty(name='test_retrieval', domain_type=DomainType.PROJECT)

# Add sample messages
messages = [
    "Victor Frankenstein was a young scientist obsessed with creating life from dead matter.",
    "The creature Frankenstein created was eight feet tall and hideously deformed.",
    "Elizabeth Lavenza was Victor's adopted sister and eventual fiancée.",
    "The Arctic explorer Robert Walton found Victor pursuing his creation across the ice.",
    "The creature learned to read and speak by observing the De Lacey family in secret.",
]

for msg in messages:
    domain.add_message(msg, metadata={'test': True})

print(f"Added {len(messages)} test messages to domain")

# Create initiative engine
boundaries = BoundariesConfig.default_luna9()
engine = InitiativeEngine(domain=domain, boundaries=boundaries)

# Test that domain query works first
test_query_result = domain.query("Victor Frankenstein", k=3)
print(f"\nDirect domain.query() test:")
print(f"  Query result keys: {list(test_query_result.keys())}")
if 'sources' in test_query_result and test_query_result['sources']:
    print(f"  Found {len(test_query_result['sources'].get('messages', []))} exact matches")
    if test_query_result['sources']['messages']:
        print(f"  First: {test_query_result['sources']['messages'][0][:80]}...")
if 'interpretation' in test_query_result and test_query_result['interpretation']:
    print(f"  Found {len(test_query_result['interpretation'].get('messages', []))} smooth results")
    if test_query_result['interpretation']['messages']:
        print(f"  First: {test_query_result['interpretation']['messages'][0][:80]}...")

# Surface context with a test query
query = "Who is Victor Frankenstein?"
print(f"\nQuery: {query}\n")

# Surface initiative context with new parameters
context = engine.surface_initiative_context(
    query=query,
    top_k=3,
    k_retrieve_per_region=2,  # Retrieve 2 messages per curvature region
    max_ambient_chars=200,    # Show up to 200 chars for ambient memories
    max_region_messages=2,    # Show 2 messages per region
    max_message_chars=150     # Show up to 150 chars per message
)

print("=" * 80)
print("FULL INITIATIVE CONTEXT:")
print("=" * 80)
print(context.full_context)
print("=" * 80)

# Verify that full text appears (not just 60-80 char previews)
if "Victor Frankenstein was a young scientist" in context.full_context:
    print("\n[PASS] Full text from ambient memories appears in context")
else:
    print("\n[FAIL] Full text not found in ambient memories")

if "Content at this junction:" in context.full_context:
    print("[PASS] Retrieved content appears in possibilities")
else:
    print("[FAIL] Retrieved content not found in possibilities")

# Check that we're not just seeing 60-char previews
if "Victor Frankenstein was a young scientist obsessed with creating" in context.full_context:
    print("[PASS] Context includes text longer than 60 chars (old preview limit)")
else:
    print("[WARNING] Text may still be truncated too aggressively")

print("\n" + "=" * 80)
print("STATE SECTION:")
print("=" * 80)
print(context.state_prose)

print("\n" + "=" * 80)
print("POSSIBILITIES SECTION:")
print("=" * 80)
print(context.possibilities_prose)

print("\nTest complete!")
