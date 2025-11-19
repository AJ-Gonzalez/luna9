"""
Demo: Baseline vs Dual Retrieval Comparison

Side-by-side comparison of traditional cosine similarity (baseline) vs
semantic surface dual retrieval on the same queries.

This is Step 1 of Phase 3 validation - qualitative assessment before
building formal metrics.
"""

from semantic_surface import SemanticSurface
from baseline import BaselineRetrieval

# The 16-message AI safety conversation
MESSAGES = [
    "What are the main risks of AGI?",
    "The main risks include misalignment, rapid capability gain, and unintended consequences.",
    "But isn't misalignment solvable with RLHF?",
    "RLHF helps but has limitations - it optimizes for human feedback, not true alignment.",
    "What about using formal verification?",
    "Formal verification is promising but currently doesn't scale to large neural networks.",
    "So what's the path forward?",
    "Multiple approaches: interpretability research, robustness testing, and iterative deployment.",
    "That sounds expensive.",
    "True, but the cost of getting it wrong is potentially catastrophic.",
    "Fair point. What about AI regulation?",
    "Regulation can help, but needs to be technically informed and internationally coordinated.",
    "Isn't that unrealistic?",
    "It's challenging but necessary - we've seen international coordination on nuclear weapons.",
    "Good analogy.",
    "Though AI is harder to regulate due to dual-use nature and rapid development."
]


def print_header(text):
    """Print a nice header."""
    print("\n" + "="*80)
    print(f"  {text}")
    print("="*80)


def print_subheader(text):
    """Print a subheader."""
    print("\n" + "-"*80)
    print(f"  {text}")
    print("-"*80)


def print_baseline_results(query, result, messages, k=3):
    """Print baseline cosine similarity results."""
    retrieved = result.get_messages(messages, k=k)

    for i, (msg, sim, idx) in enumerate(zip(
        retrieved['messages'],
        retrieved['similarities'],
        retrieved['indices']
    ), 1):
        print(f"{i}. [sim={sim:.3f}] (msg {idx})")
        print(f"   \"{msg}\"")


def print_surface_results(query, result, messages, k=3):
    """Print dual retrieval results."""
    retrieved = result.get_messages(messages, mode='both', k=k)

    print("\n  SMOOTH MODE (Interpretation):")
    for i, (msg, weight, idx) in enumerate(zip(
        retrieved['interpretation']['messages'],
        retrieved['interpretation']['weights'],
        retrieved['interpretation']['indices']
    ), 1):
        print(f"  {i}. [{weight*100:.1f}%] (msg {idx})")
        print(f"     \"{msg}\"")

    print("\n  EXACT MODE (Provenance):")
    for i, (msg, dist, idx) in enumerate(zip(
        retrieved['sources']['messages'],
        retrieved['sources']['distances'],
        retrieved['sources']['indices']
    ), 1):
        print(f"  {i}. [dist={dist:.3f}] (msg {idx})")
        print(f"     \"{msg}\"")


def compare_query(query, description, baseline, surface, messages, k=3):
    """Run comparison for a single query."""
    print_header(f"QUERY: {description}")
    print(f"\n\"{query}\"")

    # Run baseline
    print_subheader("BASELINE: Cosine Similarity")
    baseline_result = baseline.query(query, k=k)
    print_baseline_results(query, baseline_result, messages, k=k)

    # Run surface dual retrieval
    print_subheader("DUAL RETRIEVAL: Semantic Surface")
    surface_result = surface.query(query, k=k)

    # Show surface metadata
    u, v = surface_result.uv
    K, H = surface_result.curvature
    print(f"\nSurface position: (u={u:.3f}, v={v:.3f})")
    print(f"Curvature: K={K:.4f}, H={H:.4f}")

    print_surface_results(query, surface_result, messages, k=k)

    print("\n")


def main():
    """Run the comparison demo."""
    print_header("PHASE 3 VALIDATION: Baseline vs Dual Retrieval")

    print("\nSetting up both systems...")
    print("(This will download the embedding model on first run)")

    # Create both systems
    print("\n1. Creating baseline retrieval (cosine similarity)...")
    baseline = BaselineRetrieval(MESSAGES)

    print("\n2. Creating semantic surface (dual retrieval)...")
    surface = SemanticSurface(MESSAGES)

    print("\nBoth systems ready!")

    # Demo queries showing different use cases
    queries = [
        ("What are the limitations of RLHF?", "Conceptual - about ideas/sentiment"),
        ("Who mentioned nuclear weapons?", "Factual - looking for specific reference"),
        ("What are the main risks of AGI?", "Direct - matches a message exactly"),
        ("Is regulation realistic?", "Mixed - needs both interpretation and sources"),
    ]

    for query, description in queries:
        compare_query(query, description, baseline, surface, MESSAGES, k=3)

    print_header("COMPARISON COMPLETE")

    print("\nOBSERVATIONS TO MAKE:")
    print("\n  1. RETRIEVAL QUALITY:")
    print("     - Which system returns more relevant messages?")
    print("     - Does baseline miss important context that surface captures?")
    print("     - Does surface return less relevant messages than baseline?")

    print("\n  2. DIFFERENT QUERY TYPES:")
    print("     - Factual queries (\"who mentioned...\")")
    print("       Does exact mode help? Does baseline work well enough?")
    print("     - Conceptual queries (\"limitations of...\")")
    print("       Does smooth mode blend concepts better than cosine?")
    print("     - Direct matches (exact message exists)")
    print("       Both should work - which ranks it higher?")

    print("\n  3. DUAL MODES:")
    print("     - Do smooth and exact return different results?")
    print("     - Are they complementary (both useful)?")
    print("     - Could initiative engine pick the right mode?")

    print("\n  4. GEOMETRIC CONTEXT:")
    print("     - Does curvature correlate with importance?")
    print("     - Do UV positions cluster by topic?")
    print("     - Could we use this for navigation?")

    print("\nNEXT STEPS:")
    print("   - Create ground truth labels (mark relevant messages per query)")
    print("   - Implement metrics (precision, recall, F1)")
    print("   - Expand dataset (50-100 messages)")
    print("   - Run formal validation")


if __name__ == "__main__":
    main()
