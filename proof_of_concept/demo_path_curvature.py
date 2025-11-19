"""
Demo: Path Curvature as Semantic Transition Complexity

Tests the hypothesis that path curvature between messages correlates with
semantic relationship complexity.

Low curvature = direct conceptual connection
High curvature = requires intermediate concepts to bridge
"""

from semantic_surface import SemanticSurface
import numpy as np

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


def analyze_message_pair(surface, idx1, idx2, messages, relationship_type):
    """Analyze displacement and path curvature between two messages."""
    print_subheader(f"{relationship_type}: msg {idx1} ↔ msg {idx2}")

    print(f"\nMessage {idx1}: \"{messages[idx1]}\"")
    print(f"Message {idx2}: \"{messages[idx2]}\"")

    # Get UV positions for both messages
    uv1 = surface.provenance['msg_to_cp'][idx1]
    uv1 = (uv1[0] / 3.0, uv1[1] / 3.0)  # Convert to UV coordinates

    uv2 = surface.provenance['msg_to_cp'][idx2]
    uv2 = (uv2[0] / 3.0, uv2[1] / 3.0)

    print(f"\nUV positions: {uv1} → {uv2}")

    # Compute net displacement
    displacement = surface.compute_displacement(uv1, uv2)
    print(f"\nNET DISPLACEMENT:")
    print(f"  Euclidean distance: {displacement['distance']:.4f}")
    print(f"  Direction (first 5 dims): {displacement['direction'][:5]}")

    # Compute path curvature
    path_curv = surface.compute_path_curvature(uv1, uv2, num_steps=50)
    print(f"\nPATH CURVATURE (Semantic Transition Complexity):")
    print(f"  Arc length: {path_curv['arc_length']:.4f}")
    print(f"  Total curvature: {path_curv['total_curvature']:.4f} radians")
    print(f"  Max curvature: {path_curv['max_curvature']:.4f} radians")
    print(f"  Mean curvature: {path_curv['mean_curvature']:.4f} radians")

    # Compute ratio of arc length to straight-line distance
    if displacement['distance'] > 1e-10:
        path_efficiency = displacement['distance'] / path_curv['arc_length']
        print(f"  Path efficiency: {path_efficiency:.4f} (1.0 = straight line)")
    else:
        print(f"  Path efficiency: N/A (identical points)")

    return {
        'displacement': displacement,
        'path_curvature': path_curv,
        'relationship': relationship_type
    }


def main():
    """Run the path curvature analysis."""
    print_header("PATH CURVATURE: Semantic Transition Complexity Demo")

    print("\nCreating semantic surface from 16 AI safety messages...")
    surface = SemanticSurface(MESSAGES)
    print("Surface created!")

    # Test different relationship types
    results = []

    print_header("HYPOTHESIS TESTING")
    print("\nWe expect:")
    print("  - LOW curvature for direct Q&A pairs (conceptually adjacent)")
    print("  - MEDIUM curvature for related but distinct concepts")
    print("  - HIGH curvature for distant topics (need intermediate concepts)")

    # Direct Q&A pairs (should have LOW curvature)
    print_header("TEST 1: Direct Q&A Pairs (Expect LOW curvature)")

    results.append(analyze_message_pair(
        surface, 0, 1, MESSAGES,
        "Direct Q&A (question → answer)"
    ))

    results.append(analyze_message_pair(
        surface, 2, 3, MESSAGES,
        "Direct Q&A (RLHF question → answer)"
    ))

    results.append(analyze_message_pair(
        surface, 12, 13, MESSAGES,
        "Direct Q&A (regulation question → answer)"
    ))

    # Related concepts (should have MEDIUM curvature)
    print_header("TEST 2: Related Concepts (Expect MEDIUM curvature)")

    results.append(analyze_message_pair(
        surface, 1, 3, MESSAGES,
        "Related (risks → RLHF limitations)"
    ))

    results.append(analyze_message_pair(
        surface, 3, 5, MESSAGES,
        "Related (RLHF → formal verification)"
    ))

    # Distant topics (should have HIGH curvature)
    print_header("TEST 3: Distant Topics (Expect HIGH curvature)")

    results.append(analyze_message_pair(
        surface, 0, 13, MESSAGES,
        "Distant (AGI risks → nuclear weapons analogy)"
    ))

    results.append(analyze_message_pair(
        surface, 3, 15, MESSAGES,
        "Distant (RLHF → regulation challenges)"
    ))

    results.append(analyze_message_pair(
        surface, 1, 11, MESSAGES,
        "Distant (risks → regulation answer)"
    ))

    # Summary analysis
    print_header("SUMMARY ANALYSIS")

    print("\nGrouping by relationship type:\n")

    # Group results by relationship type
    qa_pairs = [r for r in results if "Direct Q&A" in r['relationship']]
    related = [r for r in results if "Related" in r['relationship']]
    distant = [r for r in results if "Distant" in r['relationship']]

    def print_stats(group, label):
        curvatures = [r['path_curvature']['total_curvature'] for r in group]
        print(f"{label}:")
        print(f"  Count: {len(curvatures)}")
        print(f"  Total curvature range: {min(curvatures):.4f} - {max(curvatures):.4f}")
        print(f"  Mean total curvature: {np.mean(curvatures):.4f}")
        print(f"  Std dev: {np.std(curvatures):.4f}")
        print()

    print_stats(qa_pairs, "Direct Q&A pairs")
    print_stats(related, "Related concepts")
    print_stats(distant, "Distant topics")

    print("\n" + "="*80)
    print("INTERPRETATION:")
    print("="*80)

    qa_mean = np.mean([r['path_curvature']['total_curvature'] for r in qa_pairs])
    related_mean = np.mean([r['path_curvature']['total_curvature'] for r in related])
    distant_mean = np.mean([r['path_curvature']['total_curvature'] for r in distant])

    print("\nIf hypothesis is correct, we should see: qa_mean < related_mean < distant_mean")
    print(f"\nActual ordering:")
    print(f"  Q&A pairs:      {qa_mean:.4f}")
    print(f"  Related:        {related_mean:.4f}")
    print(f"  Distant:        {distant_mean:.4f}")

    if qa_mean < related_mean < distant_mean:
        print("\nHYPOTHESIS CONFIRMED!")
        print("  Path curvature DOES correlate with semantic relationship complexity!")
    elif qa_mean < distant_mean:
        print("\nPARTIAL CONFIRMATION")
        print("  Direct pairs have lower curvature than distant, but related is out of order.")
    else:
        print("\nHYPOTHESIS NOT SUPPORTED")
        print("  Path curvature does not clearly correlate with relationship type.")

    print("\nPath curvature gives us semantic transition complexity!\n")


if __name__ == "__main__":
    main()
