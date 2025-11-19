"""
Test script for SemanticSurface append functionality.
"""
import sys
sys.path.insert(0, 'luna9-python')

from luna9.semantic_surface import SemanticSurface

# Initial messages (16 = 4x4 grid)
initial_messages = [
    "I love programming in Python",
    "Python has great libraries",
    "Rust is fast and safe",
    "Memory safety is important",
    "JavaScript runs in browsers",
    "TypeScript adds types to JS",
    "Go is good for concurrency",
    "Channels are Go's strength",
    "C is close to the metal",
    "Pointers are powerful in C",
    "Java has strong typing",
    "The JVM is mature",
    "Ruby is elegant",
    "Rails made Ruby popular",
    "PHP powers many websites",
    "WordPress uses PHP"
]

print("=" * 80)
print("TEST: SemanticSurface Append Functionality")
print("=" * 80)

# Create initial surface
print("\n1. Creating initial surface with 16 messages...")
surface = SemanticSurface(initial_messages)

# Query before append
print("\n2. Query BEFORE append: 'memory management'")
result = surface.query("memory management", k=3)
messages_before = result.get_messages(surface.messages, mode='exact', k=3)
print(f"   Top 3 results:")
for i, msg in enumerate(messages_before['sources']['messages'], 1):
    print(f"     {i}. {msg}")

# Append a few messages
print("\n3. Appending 4 new messages about memory...")
new_messages = [
    "Memory management is crucial in systems programming",
    "Garbage collection automates memory management",
    "Manual memory management gives more control",
    "Reference counting is another memory strategy"
]

for msg in new_messages:
    surface.append_message(msg)

print(f"   Surface now has {len(surface.messages)} messages")
print(f"   Grid size: {surface.grid_m}x{surface.grid_n}")

# Query after append
print("\n4. Query AFTER append: 'memory management'")
result2 = surface.query("memory management", k=3)
messages_after = result2.get_messages(surface.messages, mode='exact', k=3)
print(f"   Top 3 results:")
for i, msg in enumerate(messages_after['sources']['messages'], 1):
    print(f"     {i}. {msg}")

# Verify new messages appear
print("\n5. Verification:")
new_msg_texts = set(new_messages)
found_new = [msg for msg in messages_after['sources']['messages'] if msg in new_msg_texts]
print(f"   New messages in top 3: {len(found_new)}/{len(new_msg_texts)}")
if found_new:
    print(f"   SUCCESS: New messages are retrievable!")
    for msg in found_new:
        print(f"     - {msg}")
else:
    print(f"   WARNING: No new messages in top 3")

print("\n" + "=" * 80)
print("TEST COMPLETE")
print("=" * 80)
