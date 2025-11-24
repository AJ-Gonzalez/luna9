#!/usr/bin/env python3
"""Debug what domain.query() returns for single-message domains."""

import tempfile
import shutil
from luna9 import MemoryHarness

temp_dir = tempfile.mkdtemp()
print(f"Testing in: {temp_dir}\n")

try:
    harness = MemoryHarness(base_path=temp_dir)

    # Add just ONE message
    harness.remember("Test message", context="test/domain")

    # Get the domain directly
    domain = harness.manager.domains["test/domain"]

    # Call query directly
    print("Calling domain.query()...")
    result = domain.query("test", k=5, mode='both')

    print(f"\nResult type: {type(result)}")
    print(f"Result: {result}")

    if isinstance(result, dict):
        print("\n✓ It's a dict!")
        print(f"Keys: {result.keys()}")
        if 'sources' in result:
            print(f"sources type: {type(result['sources'])}")
            print(f"sources: {result['sources']}")
    else:
        print(f"\n✗ ERROR: Expected dict, got {type(result)}")

finally:
    shutil.rmtree(temp_dir)
