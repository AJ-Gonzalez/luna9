"""Debug why hash index returns no results."""

import sys
sys.path.insert(0, 'luna9-python')

import numpy as np
from luna9 import SemanticSurface, HashIndex

# Create small test surface
messages = [f"Message {i}" for i in range(16)]
surface = SemanticSurface(messages, grid_shape=(4, 4))

# Build hash index
hash_index = HashIndex(bucket_size=100, quantization_bits=8)

print("Populating hash index:")
print(f"Grid is {surface.grid_m}x{surface.grid_n}")
print("\nControl point locations:")

for i in range(surface.grid_m):
    for j in range(surface.grid_n):
        u = i / max(1, surface.grid_m - 1)
        v = j / max(1, surface.grid_n - 1)
        msg_idx = surface.provenance['cp_to_msg'][(i, j)]
        hash_val = hash_index.add_message(msg_idx, u, v)
        print(f"  CP[{i},{j}] -> u={u:.3f}, v={v:.3f}, msg_idx={msg_idx}, hash={hash_val}")

print(f"\nHash index stats: {hash_index.stats()}")
print(f"Number of buckets: {len(hash_index.buckets)}")
print(f"Bucket keys (first 10): {list(hash_index.buckets.keys())[:10]}")

# Test query at specific locations
test_points = [
    (0.0, 0.0, "Corner (0,0)"),
    (0.019, 0.072, "Query result location"),
    (0.5, 0.5, "Center"),
    (1.0, 1.0, "Corner (1,1)"),
    (0.333, 0.333, "Near (1,1) CP"),
]

print("\nTesting hash queries:")
for u, v, desc in test_points:
    # Compute what hash we'd get
    hash_val = hash_index._compute_hash(u, v)
    print(f"\n{desc}: u={u:.3f}, v={v:.3f}, hash={hash_val}")

    # Check if that hash exists
    if hash_val in hash_index.buckets:
        print(f"  ✓ Hash {hash_val} exists with {len(hash_index.buckets[hash_val])} entries")
    else:
        print(f"  ✗ Hash {hash_val} NOT in index")

    # Try querying
    entries = hash_index.query(u, v, k=3, search_radius=1)
    print(f"  Query returned {len(entries)} entries")
    for entry in entries:
        print(f"    msg_id={entry.message_id}, u={entry.u:.3f}, v={entry.v:.3f}")

    # Try with larger search radius
    entries_large = hash_index.query(u, v, k=3, search_radius=3)
    print(f"  Query with radius=3 returned {len(entries_large)} entries")
