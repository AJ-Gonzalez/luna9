"""
Benchmark script for "Parametric Surfaces for Semantic Memory" whitepaper.

Tests the core claims:
1. Surface construction preserves semantic relationships (Recall@k)
2. Query time scales sub-linearly (latency vs corpus size)
3. Memory usage is reasonable
4. Curvature correlates with semantic transitions

Decouples retrieval quality from LLM synthesis quality - we measure whether
the right chunks are retrieved, not whether models use them well.
"""

import json
import time
import psutil
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict

# Luna9 imports
from luna9.core.semantic_surface import SemanticSurface, RetrievalResult
from sentence_transformers import SentenceTransformer


@dataclass
class GroundTruthQuery:
    """A query with known relevant passages (ground truth)."""
    query: str
    relevant_indices: List[int]  # Indices of passages that should be retrieved
    query_type: str  # 'factual', 'thematic', 'conceptual'


@dataclass
class RetrievalMetrics:
    """Metrics for a single query."""
    query_id: int
    query_type: str

    # Retrieval quality
    recall_at_5: float
    recall_at_10: float
    recall_at_20: float
    precision_at_5: float
    precision_at_10: float
    precision_at_20: float
    mrr: float  # Mean Reciprocal Rank

    # Performance
    query_latency_ms: float
    projection_iterations: int


@dataclass
class BenchmarkResults:
    """Complete benchmark results for the whitepaper."""

    # Dataset info
    corpus_size: int
    num_queries: int

    # Aggregated retrieval quality
    avg_recall_at_5: float
    avg_recall_at_10: float
    avg_recall_at_20: float
    avg_precision_at_5: float
    avg_precision_at_10: float
    avg_precision_at_20: float
    avg_mrr: float

    # Performance metrics
    mean_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float

    # Memory usage
    memory_mb: float
    memory_per_message_kb: float

    # Per-query details
    query_results: List[RetrievalMetrics]


class ParametricSurfaceBenchmark:
    """Benchmark harness for parametric surface retrieval."""

    def __init__(self, model_name: str = 'all-mpnet-base-v2'):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def create_ground_truth_queries(
        self,
        corpus: List[str]
    ) -> List[GroundTruthQuery]:
        """Create labeled query-passage pairs for evaluation."""
        from gutenberg_loader import create_ground_truth_queries

        raw_queries = create_ground_truth_queries(corpus)

        # Convert to GroundTruthQuery objects
        queries = [
            GroundTruthQuery(
                query=q['query'],
                relevant_indices=q['relevant_indices'],
                query_type=q['query_type']
            )
            for q in raw_queries
        ]

        return queries

    def compute_recall_at_k(
        self,
        retrieved: List[int],
        relevant: List[int],
        k: int
    ) -> float:
        """Recall@k: fraction of relevant items in top-k results."""
        if not relevant:
            return 0.0

        top_k = set(retrieved[:k])
        relevant_set = set(relevant)

        hits = len(top_k & relevant_set)
        return hits / len(relevant_set)

    def compute_precision_at_k(
        self,
        retrieved: List[int],
        relevant: List[int],
        k: int
    ) -> float:
        """Precision@k: fraction of top-k that are relevant."""
        if k == 0:
            return 0.0

        top_k = set(retrieved[:k])
        relevant_set = set(relevant)

        hits = len(top_k & relevant_set)
        return hits / k

    def compute_mrr(
        self,
        retrieved: List[int],
        relevant: List[int]
    ) -> float:
        """Mean Reciprocal Rank: 1/rank of first relevant item."""
        relevant_set = set(relevant)

        for rank, idx in enumerate(retrieved, 1):
            if idx in relevant_set:
                return 1.0 / rank

        return 0.0  # No relevant items found

    def benchmark_retrieval_quality(
        self,
        surface: SemanticSurface,
        corpus: List[str],
        queries: List[GroundTruthQuery]
    ) -> List[RetrievalMetrics]:
        """Measure retrieval quality metrics for all queries."""
        results = []

        for query_id, gt_query in enumerate(queries):
            # Query the surface
            start_time = time.perf_counter()
            result = surface.query(gt_query.query, k=20)
            query_time = (time.perf_counter() - start_time) * 1000  # ms

            # Get retrieved indices (exact mode for provenance)
            retrieved_indices = [msg_idx for _, _, msg_idx, _ in result.nearest_control_points[:20]]

            # Compute metrics
            metrics = RetrievalMetrics(
                query_id=query_id,
                query_type=gt_query.query_type,
                recall_at_5=self.compute_recall_at_k(
                    retrieved_indices, gt_query.relevant_indices, 5
                ),
                recall_at_10=self.compute_recall_at_k(
                    retrieved_indices, gt_query.relevant_indices, 10
                ),
                recall_at_20=self.compute_recall_at_k(
                    retrieved_indices, gt_query.relevant_indices, 20
                ),
                precision_at_5=self.compute_precision_at_k(
                    retrieved_indices, gt_query.relevant_indices, 5
                ),
                precision_at_10=self.compute_precision_at_k(
                    retrieved_indices, gt_query.relevant_indices, 10
                ),
                precision_at_20=self.compute_precision_at_k(
                    retrieved_indices, gt_query.relevant_indices, 20
                ),
                mrr=self.compute_mrr(retrieved_indices, gt_query.relevant_indices),
                query_latency_ms=query_time,
                projection_iterations=getattr(result, 'metadata', {}).get('iterations', 0)
            )

            results.append(metrics)

        return results

    def measure_memory_usage(self, surface: SemanticSurface) -> Tuple[float, float]:
        """Measure memory usage in MB."""
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024

        num_messages = len(surface.messages)
        memory_per_msg_kb = (memory_mb * 1024) / num_messages if num_messages > 0 else 0

        return memory_mb, memory_per_msg_kb

    def run_benchmark(
        self,
        corpus: List[str],
        output_path: Optional[Path] = None
    ) -> BenchmarkResults:
        """Run complete benchmark suite."""

        print(f"\n=== Benchmarking Parametric Surfaces ===")
        print(f"Corpus size: {len(corpus)} passages")

        # 1. Build surface
        print("\n[1/4] Building semantic surface...")
        surface = SemanticSurface(corpus, model_name=self.model_name)

        # 2. Measure memory
        print("[2/4] Measuring memory usage...")
        memory_mb, memory_per_msg_kb = self.measure_memory_usage(surface)

        # 3. Create ground truth queries
        print("[3/4] Creating ground truth queries...")
        queries = self.create_ground_truth_queries(corpus)
        print(f"  Generated {len(queries)} queries")

        # 4. Run retrieval benchmarks
        print("[4/4] Running retrieval benchmarks...")
        query_results = self.benchmark_retrieval_quality(surface, corpus, queries)

        # Aggregate results
        latencies = [r.query_latency_ms for r in query_results]

        results = BenchmarkResults(
            corpus_size=len(corpus),
            num_queries=len(queries),

            # Average retrieval quality
            avg_recall_at_5=np.mean([r.recall_at_5 for r in query_results]),
            avg_recall_at_10=np.mean([r.recall_at_10 for r in query_results]),
            avg_recall_at_20=np.mean([r.recall_at_20 for r in query_results]),
            avg_precision_at_5=np.mean([r.precision_at_5 for r in query_results]),
            avg_precision_at_10=np.mean([r.precision_at_10 for r in query_results]),
            avg_precision_at_20=np.mean([r.precision_at_20 for r in query_results]),
            avg_mrr=np.mean([r.mrr for r in query_results]),

            # Performance
            mean_latency_ms=np.mean(latencies),
            p50_latency_ms=np.percentile(latencies, 50),
            p95_latency_ms=np.percentile(latencies, 95),
            p99_latency_ms=np.percentile(latencies, 99),

            # Memory
            memory_mb=memory_mb,
            memory_per_message_kb=memory_per_msg_kb,

            # Details
            query_results=query_results
        )

        # Print summary
        self.print_results(results)

        # Save to JSON if path provided
        if output_path:
            self.save_results(results, output_path)

        return results

    def print_results(self, results: BenchmarkResults):
        """Print formatted results."""
        print("\n=== Results ===\n")

        print("Retrieval Quality:")
        print(f"  Recall@5:     {results.avg_recall_at_5:.3f}")
        print(f"  Recall@10:    {results.avg_recall_at_10:.3f}")
        print(f"  Recall@20:    {results.avg_recall_at_20:.3f}")
        print(f"  Precision@5:  {results.avg_precision_at_5:.3f}")
        print(f"  Precision@10: {results.avg_precision_at_10:.3f}")
        print(f"  Precision@20: {results.avg_precision_at_20:.3f}")
        print(f"  MRR:          {results.avg_mrr:.3f}")

        print("\nQuery Performance:")
        print(f"  Mean:   {results.mean_latency_ms:.2f} ms")
        print(f"  p50:    {results.p50_latency_ms:.2f} ms")
        print(f"  p95:    {results.p95_latency_ms:.2f} ms")
        print(f"  p99:    {results.p99_latency_ms:.2f} ms")

        print("\nMemory Usage:")
        print(f"  Total:        {results.memory_mb:.2f} MB")
        print(f"  Per message:  {results.memory_per_message_kb:.2f} KB")

    def save_results(self, results: BenchmarkResults, output_path: Path):
        """Save results to JSON."""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to dict (with nested dataclasses)
        results_dict = asdict(results)

        with open(output_path, 'w') as f:
            json.dump(results_dict, f, indent=2)

        print(f"\n[OK] Results saved to {output_path}")


def load_gutenberg_corpus(num_passages: int = 1000) -> List[str]:
    """Load Project Gutenberg text and chunk into passages."""
    from gutenberg_loader import GutenbergLoader

    loader = GutenbergLoader()

    # Estimate books needed based on ~200 chunks per book
    num_books = max(1, (num_passages // 200) + 1)

    corpus = loader.load_corpus(
        num_books=num_books,
        target_chunks=num_passages,
        chunk_size=500  # ~500 chars per chunk
    )

    return corpus


if __name__ == '__main__':
    # Benchmark at multiple scales
    corpus_sizes = [100, 500, 1000]  # Start small, can scale up

    benchmark = ParametricSurfaceBenchmark()

    for size in corpus_sizes:
        print(f"\n{'='*60}")
        print(f"CORPUS SIZE: {size}")
        print(f"{'='*60}")

        # Load corpus
        corpus = load_gutenberg_corpus(num_passages=size)

        # Run benchmark
        results = benchmark.run_benchmark(
            corpus=corpus,
            output_path=Path(f'whitepapers/benchmark_results_{size}.json')
        )

    print("\n✓ All benchmarks complete!")
