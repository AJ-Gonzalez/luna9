"""Quick benchmark: Hash index vs full scan performance."""
import time
from luna9 import Domain, DomainType

# Create test data
messages = [f"Message {i} about topic {i % 10}" for i in range(64)]  # 8x8 grid

print("Creating domains...")
domain_with_hash = Domain.create_from_messages("bench_hash", DomainType.PROJECT, messages, use_hash_index=True)
domain_no_hash = Domain.create_from_messages("bench_no_hash", DomainType.PROJECT, messages, use_hash_index=False)

queries = ["topic 3", "message 42", "performance test", "data structures", "algorithms"]

print("\nBenchmarking WITH hash index (10 iterations)...")
start = time.perf_counter()
for _ in range(10):
    for q in queries:
        domain_with_hash.query(q, k=5)
with_hash_time = (time.perf_counter() - start) * 1000

print(f"Benchmarking WITHOUT hash index (10 iterations)...")
start = time.perf_counter()
for _ in range(10):
    for q in queries:
        domain_no_hash.query(q, k=5)
no_hash_time = (time.perf_counter() - start) * 1000

print("\n" + "="*50)
print(f"WITH hash index:    {with_hash_time:.1f}ms total, {with_hash_time/50:.2f}ms per query")
print(f"WITHOUT hash index: {no_hash_time:.1f}ms total, {no_hash_time/50:.2f}ms per query")
print(f"Speedup: {no_hash_time/with_hash_time:.2f}x faster")
print("="*50)
