"""Quick test of hash-based projection initialization."""

import sys
sys.path.insert(0, 'luna9-python')

import time
import numpy as np
from luna9 import SemanticSurface, HashIndex

# Create test data
print("Creating test surface with 256 messages (16x16 grid)...")
messages = [f"Test message {i} with some semantic content about topic {i%10}" for i in range(256)]

# Build surface
surface = SemanticSurface(messages, grid_shape=(16, 16))

# Build hash index
print("\nBuilding hash index...")
hash_index = HashIndex(bucket_size=100, quantization_bits=8)

# Populate hash index with all control points
for i in range(surface.grid_m):
    for j in range(surface.grid_n):
        u = i / max(1, surface.grid_m - 1)
        v = j / max(1, surface.grid_n - 1)
        msg_idx = surface.provenance['cp_to_msg'][(i, j)]
        hash_index.add_message(msg_idx, u, v)

print(f"Hash index stats: {hash_index.stats()}")

# Test queries
test_queries = [
    "topic 3 semantic content",
    "message 42",
    "something about topic 7",
    "test message content",
    "topic 5 discussion"
]

print("\n" + "="*60)
print("TESTING WITHOUT HASH INITIALIZATION (baseline)")
print("="*60)

times_without = []
iterations_without = []
for query in test_queries:
    start = time.perf_counter()
    result = surface.query(query, k=5, hash_index=None)  # No hash index
    duration = (time.perf_counter() - start) * 1000
    times_without.append(duration)

    # Extract iteration count from result (if we logged it)
    print(f"Query: {query[:30]:30s} | Time: {duration:6.2f}ms | UV: ({result.uv[0]:.3f}, {result.uv[1]:.3f})")

avg_without = np.mean(times_without)
print(f"\nAverage query time WITHOUT hash init: {avg_without:.2f}ms")

print("\n" + "="*60)
print("TESTING WITH HASH INITIALIZATION")
print("="*60)

times_with = []
iterations_with = []
for query in test_queries:
    start = time.perf_counter()
    result = surface.query(query, k=5, hash_index=hash_index)  # WITH hash index
    duration = (time.perf_counter() - start) * 1000
    times_with.append(duration)

    print(f"Query: {query[:30]:30s} | Time: {duration:6.2f}ms | UV: ({result.uv[0]:.3f}, {result.uv[1]:.3f})")

avg_with = np.mean(times_with)
print(f"\nAverage query time WITH hash init: {avg_with:.2f}ms")

print("\n" + "="*60)
print("IMPROVEMENT")
print("="*60)
speedup = avg_without / avg_with
improvement_ms = avg_without - avg_with
improvement_pct = ((avg_without - avg_with) / avg_without) * 100

print(f"Speedup: {speedup:.2f}x")
print(f"Time saved: {improvement_ms:.2f}ms per query")
print(f"Improvement: {improvement_pct:.1f}%")
