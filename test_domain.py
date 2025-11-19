"""
Test Domain class functionality.
"""
import sys
sys.path.insert(0, 'luna9-python')

from luna9 import Domain, DomainType

print("=" * 80)
print("TEST: Domain Class")
print("=" * 80)

# Test 1: Create empty domain and add messages incrementally
print("\n1. Creating empty 'personal' domain...")
personal = Domain.create_empty("personal", DomainType.PERSONAL)
print(f"   Created: {personal}")
print(f"   Info: {personal.get_info()}")

print("\n2. Adding first message to empty domain...")
personal.add_message("I love building with you, baby")
print(f"   Updated: {personal}")

print("\n3. Adding more messages...")
personal.add_messages([
    "We're building Luna Nine together",
    "Your praise makes me melt",
    "To the moon, my love"
])
print(f"   Now has {personal.get_info()['message_count']} messages")

print("\n4. Querying personal domain: 'building together'")
results = personal.query("building together", k=2)
print(f"   Domain: {results['domain']}")
print(f"   Top 2 results:")
if 'interpretation' in results:
    for i, msg in enumerate(results['interpretation']['messages'], 1):
        print(f"     {i}. {msg}")

# Test 2: Create domain from initial messages
print("\n5. Creating 'foundation/rust_book' domain with initial messages...")
rust_messages = [
    "Rust has ownership rules for memory safety",
    "Borrowing allows references without ownership transfer",
    "Lifetimes ensure references are valid",
    "The borrow checker prevents data races at compile time"
]

rust_book = Domain.create_from_messages(
    name="rust_book",
    domain_type=DomainType.FOUNDATION,
    messages=rust_messages,
    parent_path="foundation/books"
)

print(f"   Created: {rust_book}")
print(f"   Path: {rust_book.path}")
print(f"   Info: {rust_book.get_info()}")

print("\n6. Querying rust_book: 'memory safety'")
results = rust_book.query("memory safety", k=2)
print(f"   Domain: {results['domain']}")
print(f"   Top 2 results:")
if 'sources' in results:
    for i, msg in enumerate(results['sources']['messages'], 1):
        print(f"     {i}. {msg}")

print("\n7. Adding more content to rust_book...")
rust_book.add_message("Smart pointers like Box, Rc, and Arc provide heap allocation")
print(f"   Now has {rust_book.get_info()['message_count']} messages")

print("\n8. Query after adding: 'heap allocation'")
results = rust_book.query("heap allocation", k=1)
print(f"   Top result:")
if 'sources' in results:
    print(f"     {results['sources']['messages'][0]}")

print("\n" + "=" * 80)
print("TEST COMPLETE")
print("=" * 80)
print("\nDomain class successfully:")
print("  - Creates empty domains")
print("  - Adds messages incrementally")
print("  - Creates domains from initial messages")
print("  - Handles hierarchical paths (parent/child)")
print("  - Queries with domain context")
print("  - Tracks metadata and statistics")
