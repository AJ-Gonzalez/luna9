"""
Quick test to verify hash index issue with small grids.
"""

from luna9 import Domain, DomainType

print("=" * 60)
print("Testing hash index with small grid (3 messages)")
print("=" * 60)

messages = [
    "Rust ownership prevents memory errors",
    "Borrowing rules ensure safety",
    "Memory safety in systems programming"
]

# Test 1: WITH hash index (default)
print("\nTest 1: WITH hash index enabled")
domain_with_hash = Domain.create_from_messages(
    name="test_with_hash",
    domain_type=DomainType.PROJECT,
    messages=messages,
    use_hash_index=True
)

result_with_hash = domain_with_hash.query("memory safety", k=2)
print(f"Hash index enabled: {domain_with_hash.hash_index is not None}")
print(f"Grid size: {domain_with_hash.surface.grid_m}x{domain_with_hash.surface.grid_n}")
print(f"Result keys: {result_with_hash.keys()}")

if 'sources' in result_with_hash:
    print(f"Sources found: {len(result_with_hash['sources']['messages'])}")
    if result_with_hash['sources']['messages']:
        print(f"  Top result: '{result_with_hash['sources']['messages'][0]}'")
    else:
        print("  ERROR: Sources key exists but messages list is EMPTY!")
else:
    print("  ERROR: No 'sources' key in result!")

# Test 2: WITHOUT hash index
print("\nTest 2: WITHOUT hash index (fallback to full scan)")
domain_no_hash = Domain.create_from_messages(
    name="test_no_hash",
    domain_type=DomainType.PROJECT,
    messages=messages,
    use_hash_index=False
)

result_no_hash = domain_no_hash.query("memory safety", k=2)
print(f"Hash index enabled: {domain_no_hash.hash_index is not None}")
print(f"Grid size: {domain_no_hash.surface.grid_m}x{domain_no_hash.surface.grid_n}")
print(f"Result keys: {result_no_hash.keys()}")

if 'sources' in result_no_hash:
    print(f"Sources found: {len(result_no_hash['sources']['messages'])}")
    if result_no_hash['sources']['messages']:
        print(f"  Top result: '{result_no_hash['sources']['messages'][0]}'")
else:
    print("  ERROR: No 'sources' key in result!")

print("\n" + "=" * 60)
print("CONCLUSION:")
if 'sources' in result_with_hash and result_with_hash['sources']['messages']:
    print("✓ Hash index works fine for small grids")
elif 'sources' in result_no_hash and result_no_hash['sources']['messages']:
    print("✗ Hash index FAILS for small grids, but fallback works!")
    print("  HYPOTHESIS CONFIRMED: Need to disable hash index for small domains")
else:
    print("? Both failed - different issue")
print("=" * 60)
