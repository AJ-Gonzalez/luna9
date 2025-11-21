"""
Comprehensive Performance Benchmark for Luna Nine

Measures:
- Storage characteristics (bytes per message, compression)
- Query performance at scale (64 to 65k messages)
- Memory footprint
- Scalability curves

Usage:
    # With mock data (for development/testing)
    python benchmark_comprehensive.py --mock --max-messages 1000

    # With real parquet data (run manually to avoid token burn)
    python benchmark_comprehensive.py --parquet datasets/preprocessed_conversations.parquet
"""

import sys
import time
import json
import psutil
import os
import logging
from pathlib import Path
from typing import List, Dict, Any
import numpy as np

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('benchmark_comprehensive.log')
    ]
)
logger = logging.getLogger(__name__)

# Add luna9 to path
sys.path.insert(0, str(Path(__file__).parent / 'luna9-python'))

from luna9 import Domain, DomainType, DomainManager


class MemoryTracker:
    """Track memory usage during operations"""
    def __init__(self):
        self.process = psutil.Process()
        self.baseline = self.process.memory_info().rss / 1024 / 1024  # MB

    def current_mb(self) -> float:
        """Current memory usage in MB"""
        return self.process.memory_info().rss / 1024 / 1024

    def delta_mb(self) -> float:
        """Memory delta from baseline in MB"""
        return self.current_mb() - self.baseline


class Timer:
    """Simple timer context manager"""
    def __init__(self):
        self.start = None
        self.elapsed = None

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed = (time.perf_counter() - self.start) * 1000  # ms


def generate_mock_messages(count: int, conversation_id: str = "mock_conversation") -> List[str]:
    """
    Generate mock conversational messages similar to parquet data format.

    Creates realistic variation in length and content.
    """
    logger.debug(f"Generating {count} mock messages...")
    templates = [
        "What are your thoughts on {}?",
        "I think {} is really interesting because it {}.",
        "Can you explain more about {}?",
        "That's a good point about {}. However, {}.",
        "Let me clarify: {} means that {}.",
        "The main issue with {} is {}.",
        "I agree that {}, but we should also consider {}.",
        "From my perspective, {} seems like {}.",
        "Could you elaborate on the relationship between {} and {}?",
        "One important aspect of {} that people often miss is {}.",
    ]

    topics = [
        "machine learning", "neural networks", "semantic embeddings",
        "natural language processing", "transformer architectures",
        "attention mechanisms", "context windows", "retrieval systems",
        "vector databases", "geometric representations", "path curvature",
        "Bézier surfaces", "parametric modeling", "semantic relationships",
        "information geometry", "manifold learning", "dimensionality reduction"
    ]

    elaborations = [
        "provides better context preservation",
        "enables more efficient retrieval",
        "maintains semantic relationships",
        "scales to larger datasets",
        "reduces computational overhead",
        "preserves topological structure",
        "enables geometric reasoning",
        "supports multi-hop inference"
    ]

    messages = []
    for i in range(count):
        template = templates[i % len(templates)]
        topic1 = topics[i % len(topics)]
        topic2 = topics[(i + 1) % len(topics)]
        elaboration = elaborations[i % len(elaborations)]

        if '{}' in template:
            parts = template.count('{}')
            if parts == 1:
                msg = template.format(topic1)
            elif parts == 2:
                msg = template.format(topic1, elaboration)
            else:
                msg = template.format(topic1, topic2, elaboration)
        else:
            msg = template

        messages.append(msg)

    return messages


def load_parquet_messages(parquet_path: str, max_messages: int = None) -> List[str]:
    """Load messages from parquet file"""
    try:
        import polars as pl
    except ImportError:
        print("ERROR: polars not installed. Install with: pip install polars pyarrow")
        sys.exit(1)

    df = pl.read_parquet(parquet_path)

    # Extract text column
    messages = df['text'].to_list()

    if max_messages:
        messages = messages[:max_messages]

    print(f"Loaded {len(messages)} messages from {parquet_path}")
    return messages


# NOTE: This function is deprecated - storage measurement is now in-line in run_benchmark_suite()
# Kept for reference
# def measure_storage(...) - removed


def measure_query_performance(domain: Domain, queries: List[str], k: int = 5) -> Dict[str, Any]:
    """Measure query performance with cold and warm runs"""

    timings = {
        'cold_start_ms': [],
        'warm_query_ms': [],
        'queries_tested': len(queries)
    }

    for i, query in enumerate(queries):
        # Cold start (first query or after clearing cache)
        print(f"    Query {i+1}/{len(queries)}: ", end='', flush=True)
        with Timer() as t:
            result = domain.query(query, k=k, mode='both')

        if i == 0:
            timings['cold_start_ms'] = t.elapsed
            print(f"{t.elapsed:.1f}ms (cold)")
        else:
            timings['warm_query_ms'].append(t.elapsed)
            print(f"{t.elapsed:.1f}ms")

    # Calculate statistics for warm queries
    if timings['warm_query_ms']:
        timings['warm_mean_ms'] = np.mean(timings['warm_query_ms'])
        timings['warm_p50_ms'] = np.percentile(timings['warm_query_ms'], 50)
        timings['warm_p95_ms'] = np.percentile(timings['warm_query_ms'], 95)
        timings['warm_p99_ms'] = np.percentile(timings['warm_query_ms'], 99)
        timings['warm_min_ms'] = np.min(timings['warm_query_ms'])
        timings['warm_max_ms'] = np.max(timings['warm_query_ms'])

    # Remove raw data to save space in output
    del timings['warm_query_ms']

    return timings


def measure_memory_footprint(domain: Domain, tracker: MemoryTracker) -> Dict[str, Any]:
    """Measure memory footprint"""

    return {
        'domain_loaded_mb': tracker.delta_mb(),
        'total_process_mb': tracker.current_mb()
    }


def run_benchmark_suite(messages: List[str], scale_name: str, output_dir: Path) -> Dict[str, Any]:
    """Run complete benchmark suite for a given message scale"""

    logger.info(f"{'='*60}")
    logger.info(f"Benchmarking: {scale_name} ({len(messages)} messages)")
    logger.info(f"{'='*60}")

    tracker = MemoryTracker()
    results = {
        'scale': scale_name,
        'message_count': len(messages),
        'baseline_memory_mb': tracker.baseline
    }

    # Measure domain creation time (in-memory, no persistence)
    print(f"Creating domain...")
    print(f"  [1/4] Embedding {len(messages)} messages...", flush=True)

    t_total_start = time.perf_counter()

    with Timer() as t:
        domain = Domain.create_from_messages(
            name=scale_name,
            domain_type=DomainType.PROJECT,
            messages=messages
        )

    results['creation_time_ms'] = t.elapsed
    print(f"  > Total creation: {t.elapsed:.1f}ms")

    # Break down where time went (estimated from surface shape)
    if domain.surface:
        print(f"  > Surface shape: {domain.surface.control_points.shape[0]}×{domain.surface.control_points.shape[1]}")
        print(f"  > Control points: {domain.surface.control_points.shape[0] * domain.surface.control_points.shape[1]}")

    # Estimate in-memory size (Domain doesn't auto-persist to disk)
    print(f"  [2/4] Measuring storage...", flush=True)
    import sys
    surface_size = sys.getsizeof(domain.surface.control_points) if domain.surface else 0
    embeddings_size = sys.getsizeof(domain.surface.embeddings) if domain.surface else 0
    messages_size = sum(sys.getsizeof(m) for m in messages)

    results['storage'] = {
        'surface_bytes': surface_size,
        'embeddings_bytes': embeddings_size,
        'messages_bytes': messages_size,
        'total_bytes': surface_size + embeddings_size + messages_size,
        'bytes_per_message': (surface_size + embeddings_size) / len(messages) if len(messages) > 0 else 0,
        'message_count': len(messages),
        'surface_shape': domain.surface.control_points.shape if domain.surface else None
    }
    print(f"  > Storage: {results['storage']['total_bytes']:,} bytes ({results['storage']['bytes_per_message']:.1f} bytes/msg)")

    # Measure memory after loading
    print(f"  [3/4] Measuring memory footprint...", flush=True)
    results['memory'] = measure_memory_footprint(domain, tracker)
    print(f"  > Memory: {results['memory']['domain_loaded_mb']:.1f} MB delta, {results['memory']['total_process_mb']:.1f} MB total")

    # Generate test queries (from actual messages)
    num_queries = min(10, len(messages))
    test_queries = messages[::len(messages)//num_queries][:num_queries]

    # Measure query performance
    print(f"  [4/4] Running {num_queries} queries (watch for slow projection on large surfaces)...", flush=True)
    results['query_performance'] = measure_query_performance(domain, test_queries, k=5)
    print(f"  > Cold start: {results['query_performance']['cold_start_ms']:.1f}ms")
    if 'warm_mean_ms' in results['query_performance']:
        print(f"  > Warm mean: {results['query_performance']['warm_mean_ms']:.1f}ms " +
              f"(p95: {results['query_performance']['warm_p95_ms']:.1f}ms)")

    # Show total benchmark time
    total_time_s = (time.perf_counter() - t_total_start)
    print(f"  > Total benchmark time: {total_time_s:.1f}s")

    # No cleanup needed - domain only exists in memory

    return results


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Benchmark Luna Nine performance')
    parser.add_argument('--mock', action='store_true', help='Use mock data')
    parser.add_argument('--parquet', type=str, help='Path to parquet file')
    parser.add_argument('--max-messages', type=int, help='Limit number of messages')
    parser.add_argument('--output', type=str, default='benchmark_results.json',
                       help='Output JSON file')

    args = parser.parse_args()

    # Setup output directory
    output_dir = Path(__file__).parent / 'benchmark_output'
    output_dir.mkdir(exist_ok=True)

    # Load or generate messages
    if args.parquet:
        messages_full = load_parquet_messages(args.parquet, args.max_messages)
    elif args.mock:
        max_msg = args.max_messages or 1000
        print(f"Generating {max_msg} mock messages...")
        messages_full = generate_mock_messages(max_msg)
    else:
        print("ERROR: Must specify --mock or --parquet")
        sys.exit(1)

    # Define scales to test
    scales = []

    if len(messages_full) >= 64:
        scales.append(('64_messages', messages_full[:64]))
    if len(messages_full) >= 256:
        scales.append(('256_messages', messages_full[:256]))
    if len(messages_full) >= 1024:
        scales.append(('1k_messages', messages_full[:1024]))
    if len(messages_full) >= 4096:
        scales.append(('4k_messages', messages_full[:4096]))
    if len(messages_full) >= 16384:
        scales.append(('16k_messages', messages_full[:16384]))
    if len(messages_full) >= 65536:
        scales.append(('65k_messages', messages_full[:65536]))

    # If we have fewer messages, just test what we have
    if len(messages_full) < 64:
        scales.append((f'{len(messages_full)}_messages', messages_full))

    # Run benchmarks
    all_results = []

    print(f"\n{'='*60}")
    print(f"LUNA NINE COMPREHENSIVE BENCHMARK")
    print(f"{'='*60}")
    print(f"Total messages available: {len(messages_full)}")
    print(f"Scales to test: {len(scales)}")
    print(f"Output: {args.output}")

    for scale_name, messages in scales:
        try:
            result = run_benchmark_suite(messages, scale_name, output_dir)
            all_results.append(result)
        except Exception as e:
            print(f"\nERROR benchmarking {scale_name}: {e}")
            import traceback
            traceback.print_exc()

    # Save results
    output_path = Path(args.output)
    with open(output_path, 'w') as f:
        json.dump({
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_messages_available': len(messages_full),
            'scales_tested': len(all_results),
            'results': all_results
        }, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Benchmark complete!")
    print(f"Results saved to: {output_path}")
    print(f"{'='*60}\n")

    # Print summary
    print("\nSUMMARY:")
    print(f"{'Scale':<20} {'Messages':<10} {'Storage (KB)':<15} {'Cold (ms)':<12} {'Warm P95 (ms)':<15}")
    print("-" * 75)

    for r in all_results:
        storage_kb = r['storage']['total_bytes'] / 1024
        cold_ms = r['query_performance']['cold_start_ms']
        warm_p95 = r['query_performance'].get('warm_p95_ms', 'N/A')
        warm_str = f"{warm_p95:.1f}" if isinstance(warm_p95, (int, float)) else warm_p95

        print(f"{r['scale']:<20} {r['message_count']:<10} {storage_kb:<15.1f} " +
              f"{cold_ms:<12.1f} {warm_str:<15}")


if __name__ == '__main__':
    main()
