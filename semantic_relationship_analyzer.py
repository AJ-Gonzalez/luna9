"""
Semantic Relationship Analyzer for Product Positioning

Uses geometric inference (direct embeddings or surface analysis) to analyze
relationships between products, features, or market positions.

Returns structured data for LLM interpretation with zero preprocessing bias.

Usage:
    from semantic_relationship_analyzer import analyze_relationships

    products = [
        "Meeting notes with AI summarization",
        "Notes for meetings with bullet points",
        "Meeting notes visualized as diagrams",
        "Voice recordings transcribed to text",
        "Collaborative note-taking with sync"
    ]

    analysis = analyze_relationships(products)
    # Feed analysis to LLM for human-readable insights
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'luna9-python'))

from typing import List, Dict, Any
import numpy as np
from sentence_transformers import SentenceTransformer


def analyze_relationships(
    items: List[str],
    use_surface: bool = None  # None = auto-detect based on item count
) -> Dict[str, Any]:
    """
    Analyze semantic relationships using pure geometric inference.

    Args:
        items: List of text descriptions (products, features, etc.)
        use_surface: Force surface analysis (True) or embeddings (False)
                    None = auto (surface if >= 10 items, embeddings otherwise)

    Returns:
        Dictionary with:
        - items: Original items with indices
        - pairwise: All pairwise relationships
        - individual: Per-item analysis
        - summary: High-level insights
        - method: 'embeddings' or 'surface'
    """

    if len(items) < 2:
        raise ValueError("Need at least 2 items to analyze relationships")

    # Auto-detect method
    if use_surface is None:
        use_surface = len(items) >= 10

    # Get embeddings
    print(f"Embedding {len(items)} items...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(items, show_progress_bar=False)

    if use_surface:
        return _analyze_with_surface(items, embeddings)
    else:
        return _analyze_with_embeddings(items, embeddings)


def _analyze_with_embeddings(items: List[str], embeddings: np.ndarray) -> Dict[str, Any]:
    """Direct embedding comparison for small item counts."""

    print("Using direct embedding analysis...")

    # Normalize embeddings for cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings_norm = embeddings / norms

    # Compute all pairwise relationships
    print("Computing pairwise relationships...")
    pairwise = []
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            # Cosine similarity (dot product of normalized vectors)
            cosine_sim = float(np.dot(embeddings_norm[i], embeddings_norm[j]))

            # Euclidean distance in embedding space
            euclidean_dist = float(np.linalg.norm(embeddings[i] - embeddings[j]))

            # Angular distance (0 to π radians)
            # More interpretable than cosine sim
            angular_dist = float(np.arccos(np.clip(cosine_sim, -1.0, 1.0)))

            # Relationship strength (inverse of angular distance)
            strength = 1.0 / (1.0 + angular_dist)

            pairwise.append({
                'item_a_idx': i,
                'item_b_idx': j,
                'item_a': items[i],
                'item_b': items[j],
                'cosine_similarity': cosine_sim,
                'angular_distance': angular_dist,
                'euclidean_distance': euclidean_dist,
                'relationship_strength': strength,
                'interpretation': _interpret_embedding_relationship(cosine_sim, angular_dist)
            })

    # Per-item analysis
    print("Analyzing individual items...")
    individual = []
    for i in range(len(items)):
        # Find most similar and most different items
        similarities = []
        for j in range(len(items)):
            if i != j:
                cos_sim = float(np.dot(embeddings_norm[i], embeddings_norm[j]))
                similarities.append((j, cos_sim))

        similarities.sort(key=lambda x: x[1], reverse=True)
        most_similar = similarities[0] if similarities else None
        least_similar = similarities[-1] if similarities else None

        # Compute "uniqueness" as average distance to all other items
        # High uniqueness = far from everything else
        if len(items) > 2:
            avg_similarity = np.mean([s[1] for s in similarities])
            uniqueness = 1.0 - avg_similarity  # Convert to distance-like metric
        else:
            uniqueness = 0.5

        individual.append({
            'idx': i,
            'item': items[i],
            'uniqueness_score': float(uniqueness),
            'uniqueness': _interpret_uniqueness(uniqueness),
            'most_similar_to': {
                'idx': most_similar[0],
                'item': items[most_similar[0]],
                'similarity': most_similar[1]
            } if most_similar else None,
            'least_similar_to': {
                'idx': least_similar[0],
                'item': items[least_similar[0]],
                'similarity': least_similar[1]
            } if least_similar else None
        })

    # Generate summary
    summary = _generate_embedding_summary(pairwise, individual)

    return {
        'items': [{'idx': i, 'text': item} for i, item in enumerate(items)],
        'pairwise_relationships': pairwise,
        'individual_analysis': individual,
        'summary': summary,
        'metadata': {
            'num_items': len(items),
            'method': 'embeddings',
            'embedding_dim': embeddings.shape[1]
        }
    }


def _analyze_with_surface(items: List[str], embeddings: np.ndarray) -> Dict[str, Any]:
    """Surface-based analysis for larger item counts."""

    from luna9 import SemanticSurface
    from luna9.surface_math import geodesic_distance, compute_curvature, compute_path_curvature

    print("Using surface-based analysis...")
    surface = SemanticSurface(items, embeddings=embeddings)

    # Project each item onto surface
    print("Projecting items onto surface...")
    projections = []
    for i in range(len(items)):
        u, v = surface.project_embedding(embeddings[i])
        projections.append((u, v))

    # Compute pairwise relationships
    print("Computing pairwise relationships...")
    pairwise = []
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            u1, v1 = projections[i]
            u2, v2 = projections[j]

            geo_dist = geodesic_distance(
                surface.control_points, surface.weights,
                (u1, v1), (u2, v2), num_steps=50
            )

            path_curv_data = compute_path_curvature(
                surface.control_points, surface.weights,
                (u1, v1), (u2, v2), num_steps=50
            )

            strength = 1.0 / (1.0 + geo_dist)

            pairwise.append({
                'item_a_idx': i,
                'item_b_idx': j,
                'item_a': items[i],
                'item_b': items[j],
                'geodesic_distance': float(geo_dist),
                'path_curvature': float(path_curv_data['total_curvature']),
                'relationship_strength': float(strength),
                'interpretation': _interpret_surface_relationship(geo_dist, path_curv_data['total_curvature'])
            })

    # Per-item analysis
    individual = []
    for i, (u, v) in enumerate(projections):
        K, H = compute_curvature(surface.control_points, surface.weights, u, v)

        individual.append({
            'idx': i,
            'item': items[i],
            'position': {'u': float(u), 'v': float(v)},
            'gaussian_curvature': float(K),
            'uniqueness': _interpret_curvature(K)
        })

    summary = _generate_surface_summary(pairwise, individual)

    return {
        'items': [{'idx': i, 'text': item} for i, item in enumerate(items)],
        'pairwise_relationships': pairwise,
        'individual_analysis': individual,
        'summary': summary,
        'metadata': {
            'num_items': len(items),
            'method': 'surface',
            'surface_shape': f"{surface.grid_m}x{surface.grid_n}"
        }
    }


def _interpret_embedding_relationship(cosine_sim: float, angular_dist: float) -> str:
    """Interpret relationship from embedding metrics."""
    if cosine_sim > 0.9:
        return "very_similar"
    elif cosine_sim > 0.7:
        return "similar"
    elif cosine_sim > 0.5:
        return "moderately_related"
    elif cosine_sim > 0.3:
        return "somewhat_related"
    else:
        return "different"


def _interpret_surface_relationship(geo_dist: float, path_curv: float) -> str:
    """Interpret relationship from surface metrics."""
    if geo_dist < 0.1:
        dist = "very_close"
    elif geo_dist < 0.3:
        dist = "close"
    else:
        dist = "distant"

    if path_curv < 0.05:
        curv = "direct"
    else:
        curv = "indirect"

    return f"{dist}_{curv}"


def _interpret_uniqueness(uniqueness: float) -> str:
    """Interpret uniqueness score."""
    if uniqueness > 0.6:
        return "highly_unique_position"
    elif uniqueness > 0.4:
        return "moderately_unique"
    elif uniqueness > 0.2:
        return "somewhat_crowded_space"
    else:
        return "very_crowded_space"


def _interpret_curvature(K: float) -> str:
    """Interpret Gaussian curvature."""
    if abs(K) < 0.01:
        return "flat_crowded_space"
    elif K > 0.05:
        return "peak_unique_position"
    else:
        return "moderate_differentiation"


def _generate_embedding_summary(pairwise: List[Dict], individual: List[Dict]) -> Dict[str, Any]:
    """Generate summary for embedding-based analysis."""

    most_similar = max(pairwise, key=lambda x: x['cosine_similarity'])
    least_similar = min(pairwise, key=lambda x: x['cosine_similarity'])
    most_unique = max(individual, key=lambda x: x['uniqueness_score'])

    return {
        'most_similar_pair': {
            'items': [most_similar['item_a'], most_similar['item_b']],
            'similarity': most_similar['cosine_similarity'],
            'interpretation': most_similar['interpretation']
        },
        'most_different_pair': {
            'items': [least_similar['item_a'], least_similar['item_b']],
            'similarity': least_similar['cosine_similarity'],
            'interpretation': least_similar['interpretation']
        },
        'most_unique_item': {
            'item': most_unique['item'],
            'uniqueness_score': most_unique['uniqueness_score'],
            'interpretation': most_unique['uniqueness']
        },
        'avg_similarity': float(np.mean([r['cosine_similarity'] for r in pairwise])),
        'avg_angular_distance': float(np.mean([r['angular_distance'] for r in pairwise]))
    }


def _generate_surface_summary(pairwise: List[Dict], individual: List[Dict]) -> Dict[str, Any]:
    """Generate summary for surface-based analysis."""

    closest = min(pairwise, key=lambda x: x['geodesic_distance'])
    farthest = max(pairwise, key=lambda x: x['geodesic_distance'])
    most_unique = max(individual, key=lambda x: abs(x['gaussian_curvature']))

    return {
        'closest_pair': {
            'items': [closest['item_a'], closest['item_b']],
            'distance': closest['geodesic_distance']
        },
        'farthest_pair': {
            'items': [farthest['item_a'], farthest['item_b']],
            'distance': farthest['geodesic_distance']
        },
        'most_unique_item': {
            'item': most_unique['item'],
            'curvature': most_unique['gaussian_curvature']
        }
    }


def format_for_llm(analysis: Dict[str, Any]) -> str:
    """Format analysis as LLM-readable text."""

    lines = []
    lines.append("=== SEMANTIC RELATIONSHIP ANALYSIS ===\n")
    lines.append(f"Analysis Method: {analysis['metadata']['method'].upper()}\n")

    lines.append("ITEMS ANALYZED:")
    for item in analysis['items']:
        lines.append(f"  [{item['idx']}] {item['text']}")
    lines.append("")

    lines.append("PAIRWISE RELATIONSHIPS:")
    for rel in analysis['pairwise_relationships']:
        lines.append(f"  [{rel['item_a_idx']}] ↔ [{rel['item_b_idx']}]:")
        if 'cosine_similarity' in rel:
            lines.append(f"    Cosine Similarity: {rel['cosine_similarity']:.3f}")
            lines.append(f"    Angular Distance: {rel['angular_distance']:.3f} rad")
        else:
            lines.append(f"    Geodesic Distance: {rel['geodesic_distance']:.3f}")
            lines.append(f"    Path Curvature: {rel['path_curvature']:.3f}")
        lines.append(f"    Relationship: {rel['interpretation']}")
        lines.append("")

    lines.append("INDIVIDUAL ANALYSIS:")
    for item in analysis['individual_analysis']:
        lines.append(f"  [{item['idx']}] {item['item']}")
        if 'uniqueness_score' in item:
            lines.append(f"    Uniqueness: {item['uniqueness_score']:.3f} ({item['uniqueness']})")
            if item['most_similar_to']:
                lines.append(f"    Most similar to: [{item['most_similar_to']['idx']}] (sim: {item['most_similar_to']['similarity']:.3f})")
        else:
            lines.append(f"    Position: ({item['position']['u']:.3f}, {item['position']['v']:.3f})")
            lines.append(f"    Curvature: {item['gaussian_curvature']:.4f} ({item['uniqueness']})")
        lines.append("")

    lines.append("KEY INSIGHTS:")
    summary = analysis['summary']
    if 'most_similar_pair' in summary:
        lines.append(f"  Most Similar: {summary['most_similar_pair']['items'][0]} ↔ {summary['most_similar_pair']['items'][1]}")
        lines.append(f"    Similarity: {summary['most_similar_pair']['similarity']:.3f}")
    if 'most_unique_item' in summary:
        lines.append(f"  Most Unique: {summary['most_unique_item']['item']}")
        if 'uniqueness_score' in summary['most_unique_item']:
            lines.append(f"    Score: {summary['most_unique_item']['uniqueness_score']:.3f}")

    return "\n".join(lines)


if __name__ == "__main__":
    # Example: SaaS product positioning
    products = [
        "Meeting notes with AI-powered summarization and action items",
        "Note-taking for meetings with automatic bullet point formatting",
        "Meeting transcription converted into visual mind maps and diagrams",
        "Voice-to-text meeting recordings with speaker identification",
        "Real-time collaborative note-taking with team synchronization"
    ]

    print("Analyzing product relationships...\n")
    analysis = analyze_relationships(products)

    print(format_for_llm(analysis))

    import json
    with open('relationship_analysis.json', 'w') as f:
        json.dump(analysis, f, indent=2)
    print("\nFull analysis saved to: relationship_analysis.json")
