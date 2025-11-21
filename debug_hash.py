"""Debug hash index integration."""

import sys
sys.path.insert(0, 'luna9-python')

import numpy as np
from luna9 import SemanticSurface, HashIndex

# Create small test surface
print("Creating test surface...")
messages = [f"Message {i}" for i in range(16)]
surface = SemanticSurface(messages, grid_shape=(4, 4))

# Build hash index
hash_index = HashIndex(bucket_size=100, quantization_bits=8)
for i in range(surface.grid_m):
    for j in range(surface.grid_n):
        u = i / max(1, surface.grid_m - 1)
        v = j / max(1, surface.grid_n - 1)
        msg_idx = surface.provenance['cp_to_msg'][(i, j)]
        hash_index.add_message(msg_idx, u, v)

print(f"Hash index populated with {hash_index.stats()['total_messages']} messages")

# Test query
query = "test message"
print(f"\nQuerying: '{query}'")

result = surface.query(query, k=3, hash_index=hash_index)
print(f"Result UV: {result.uv}")
print(f"Nearest control points: {result.nearest_control_points}")

# Check what get_messages returns
retrieved = result.get_messages(surface.messages, mode='exact', k=3)
print(f"\nRetrieved sources: {retrieved.get('sources')}")

# Manually check nearest_control_points
if result.nearest_control_points:
    print("\nManually extracting messages from nearest_control_points:")
    for i, j, msg_idx, dist in result.nearest_control_points[:3]:
        print(f"  CP[{i},{j}] -> msg_idx={msg_idx}, dist={dist:.4f}, msg='{messages[msg_idx]}'")
else:
    print("\n⚠ nearest_control_points is EMPTY!")

# Test hash index query directly
print(f"\nTesting hash index query at UV {result.uv}:")
entries = hash_index.query(result.uv[0], result.uv[1], k=3)
print(f"Hash index returned {len(entries)} entries:")
for entry in entries:
    print(f"  msg_id={entry.message_id}, u={entry.u:.3f}, v={entry.v:.3f}, dist={entry.distance_to(result.uv[0], result.uv[1]):.4f}")
