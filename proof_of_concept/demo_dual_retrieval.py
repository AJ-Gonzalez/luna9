"""
Demo: Dual Retrieval System in Action

Shows semantic surface with both smooth (interpretation) and exact (provenance)
retrieval working together.

Run this to see your moonshot respond to real queries!
"""

from semantic_surface import SemanticSurface

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
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70)


def print_results(query, result, messages, k=3):
    """Pretty print dual retrieval results."""
    print(f"\nQuery: \"{query}\"")
    print(f"Position on surface: (u={result.uv[0]:.3f}, v={result.uv[1]:.3f})")
    K, H = result.curvature
    print(f"Curvature: K={K:.4f}, H={H:.4f}")

    # Get messages in both modes
    retrieved = result.get_messages(messages, mode='both', k=k)

    print("\n" + "-"*70)
    print("SMOOTH RETRIEVAL (Interpretation - weighted blend)")
    print("-"*70)

    for i, (msg, weight, idx) in enumerate(zip(
        retrieved['interpretation']['messages'],
        retrieved['interpretation']['weights'],
        retrieved['interpretation']['indices']
    ), 1):
        print(f"{i}. [{weight*100:.1f}%] (msg {idx})")
        print(f"   \"{msg}\"")

    print("\n" + "-"*70)
    print("EXACT RETRIEVAL (Sources - nearest control points)")
    print("-"*70)

    for i, (msg, dist, idx) in enumerate(zip(
        retrieved['sources']['messages'],
        retrieved['sources']['distances'],
        retrieved['sources']['indices']
    ), 1):
        print(f"{i}. [dist={dist:.3f}] (msg {idx})")
        print(f"   \"{msg}\"")


def main():
    """Run the demo."""
    print_header("SEMANTIC SURFACE: DUAL RETRIEVAL DEMO")

    print("\nCreating semantic surface from 16 AI safety messages...")
    print("(This will download the embedding model on first run)")

    # Create surface with real embeddings
    surface = SemanticSurface(MESSAGES)

    print("\nSurface created! Ready for queries.")

    # Demo queries showing different use cases
    queries = [
        ("What are the limitations of RLHF?", "Conceptual - about ideas/sentiment"),
        ("Who mentioned nuclear weapons?", "Factual - looking for specific reference"),
        ("What are the main risks of AGI?", "Direct - matches a message exactly"),
        ("Is regulation realistic?", "Mixed - needs both interpretation and sources"),
    ]

    for query, description in queries:
        print_header(f"QUERY: {description}")

        # Run dual retrieval
        result = surface.query(query, k=3)

        # Display results
        print_results(query, result, MESSAGES, k=3)

        print("\n")

    print_header("DEMO COMPLETE")
    print("\nNotice how:")
    print("  - Smooth retrieval shows WEIGHTED BLENDING of concepts")
    print("  - Exact retrieval shows NEAREST SOURCE messages")
    print("  - Same query, different modes, complementary results")
    print("  - Initiative engine could decide which to use based on query type")
    print("\nThis is 'thinking with a book in hand' - semantic navigation + exact provenance!")


if __name__ == "__main__":
    main()
