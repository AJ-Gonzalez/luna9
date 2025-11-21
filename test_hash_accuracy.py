"""Test if hash-based initialization affects retrieval quality."""

import sys
sys.path.insert(0, 'luna9-python')

import numpy as np
from luna9 import SemanticSurface, HashIndex

# Create test data with clear semantic structure
print("Creating test surface with 64 messages (8x8 grid)...")
messages = []
topics = ["machine learning", "cooking recipes", "space exploration", "gardening tips"]
for i in range(64):
    topic = topics[i % len(topics)]
    messages.append(f"Message {i} about {topic} with details and information")

# Build surface
surface = SemanticSurface(messages, grid_shape=(8, 8))

# Build hash index
hash_index = HashIndex(bucket_size=100, quantization_bits=8)
for i in range(surface.grid_m):
    for j in range(surface.grid_n):
        u = i / max(1, surface.grid_m - 1)
        v = j / max(1, surface.grid_n - 1)
        msg_idx = surface.provenance['cp_to_msg'][(i, j)]
        hash_index.add_message(msg_idx, u, v)

# Test queries for each topic
test_queries = [
    ("What are machine learning techniques?", "machine learning"),
    ("How do I make a cake?", "cooking recipes"),
    ("Tell me about Mars exploration", "space exploration"),
    ("Best practices for growing tomatoes", "gardening tips")
]

print("\n" + "="*80)
print("COMPARING RETRIEVAL QUALITY")
print("="*80)

for query, expected_topic in test_queries:
    print(f"\nQuery: '{query}'")
    print(f"Expected topic: {expected_topic}")

    # WITHOUT hash init
    result_without = surface.query(query, k=3, hash_index=None)
    retrieved_without = result_without.get_messages(surface.messages, mode='exact', k=3)

    print("\n  WITHOUT hash init:")
    print(f"    UV: ({result_without.uv[0]:.3f}, {result_without.uv[1]:.3f})")
    print("    Top 3 messages:")
    for msg in retrieved_without['sources']['messages']:
        print(f"      - {msg[:60]}")

    # WITH hash init
    result_with = surface.query(query, k=3, hash_index=hash_index)
    retrieved_with = result_with.get_messages(surface.messages, mode='exact', k=3)

    print("\n  WITH hash init:")
    print(f"    UV: ({result_with.uv[0]:.3f}, {result_with.uv[1]:.3f})")
    print("    Top 3 messages:")
    for msg in retrieved_with['sources']['messages']:
        print(f"      - {msg[:60]}")

    # Check if both retrieved the expected topic
    without_correct = any(expected_topic in msg for msg in retrieved_without['sources']['messages'][:3])
    with_correct = any(expected_topic in msg for msg in retrieved_with['sources']['messages'][:3])

    print(f"\n  ✓ WITHOUT retrieved correct topic: {without_correct}")
    print(f"  ✓ WITH retrieved correct topic: {with_correct}")

    if without_correct and with_correct:
        print("  → Both methods work correctly!")
    elif without_correct and not with_correct:
        print("  ⚠ Hash init may have found worse solution")
    elif not without_correct and with_correct:
        print("  ⚠ Only hash init found correct solution (interesting!)")
    else:
        print("  ⚠ Neither method retrieved expected topic")

print("\n" + "="*80)
