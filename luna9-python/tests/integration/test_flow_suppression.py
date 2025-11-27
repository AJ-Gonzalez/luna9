#!/usr/bin/env python3
"""
EXPERIMENT: Flow Suppression to Reveal Dispersed Signals

Hypothesis:
- Cohesive content (Felix) creates strong "flow" on the surface
- Dispersed content (Elizabeth) gets drowned out by that flow
- If we SUPPRESS the strong flow, the weak dispersed signals become visible

Like audio compression/limiting - bring down the loud parts so you can hear the quiet parts.

Let's try 3 different suppression functions and see what happens.
"""

import sys
from pathlib import Path
import numpy as np
from collections import Counter
import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag

sys.path.insert(0, str(Path(__file__).parent))

from luna9.core.semantic_surface import SemanticSurface


def chunk_text(text: str, chunk_size: int = 1000) -> list:
    """Same chunking as benchmark."""
    paragraphs = text.split('\n\n')
    chunks = []
    current_chunk = []
    current_size = 0

    for para in paragraphs:
        para_size = len(para)
        if current_size + para_size > chunk_size and current_chunk:
            chunks.append('\n\n'.join(current_chunk))
            current_chunk = [para]
            current_size = para_size
        else:
            current_chunk.append(para)
            current_size += para_size

    if current_chunk:
        chunks.append('\n\n'.join(current_chunk))
    return chunks


def compute_surface_normals(surface):
    """
    Compute normal vectors at each control point.

    Normal = perpendicular to tangent vectors (∂S/∂u and ∂S/∂v)
    In 768D space, there are 766 normal dimensions.

    For simplicity, we'll approximate using curvature:
    - High curvature = strong directional flow
    - Low curvature = weak/dispersed
    """
    normals = []
    magnitudes = []

    for i in range(surface.grid_m):
        for j in range(surface.grid_n):
            u = i / max(1, surface.grid_m - 1)
            v = j / max(1, surface.grid_n - 1)

            # We'll compute curvature differently - using the embedding deviation
            # (Curvature isn't needed for this calculation)

            # Get the actual embedding at this control point
            msg_idx = surface.provenance['cp_to_msg'][(i, j)]
            embedding = surface.embeddings[msg_idx]

            # Surface point at (u,v)
            surface_point = surface.evaluate_at(u, v)

            # "Normal" = difference between actual embedding and surface interpolation
            # This represents the 766D orthogonal component
            normal = embedding - surface_point
            magnitude = np.linalg.norm(normal)

            normals.append(normal)
            magnitudes.append(magnitude)

    return np.array(normals), np.array(magnitudes)


def suppression_exponential(magnitude, alpha=2.0):
    """
    Exponential suppression: exp(alpha * m) - 1
    Strong flows get suppressed exponentially.
    """
    return np.exp(alpha * magnitude) - 1


def suppression_power(magnitude, beta=3.0):
    """
    Power law suppression: m^beta
    beta > 1 means stronger suppression of strong signals.
    """
    return magnitude ** beta


def suppression_sigmoid(magnitude, gamma=5.0):
    """
    Sigmoid-based: m * sigmoid(gamma * m)
    Smooth transition from preserving weak to suppressing strong.
    """
    sigmoid = 1 / (1 + np.exp(-gamma * magnitude))
    return magnitude * sigmoid


def apply_flow_suppression(surface, normals, magnitudes, suppression_fn, fn_name):
    """
    Apply flow suppression and see what signals emerge.
    """
    print(f"\n{'='*70}")
    print(f"TESTING: {fn_name}")
    print(f"{'='*70}\n")

    # Compute suppression factors
    suppression_factors = suppression_fn(magnitudes)

    print(f"Magnitude stats:")
    print(f"  Min: {magnitudes.min():.4f}")
    print(f"  Max: {magnitudes.max():.4f}")
    print(f"  Mean: {magnitudes.mean():.4f}")

    print(f"\nSuppression factor stats:")
    print(f"  Min: {suppression_factors.min():.4f}")
    print(f"  Max: {suppression_factors.max():.4f}")
    print(f"  Mean: {suppression_factors.mean():.4f}")

    # Suppress the flow
    # For each normal, reduce it by its suppression factor
    suppressed_normals = normals.copy()
    for i in range(len(normals)):
        if magnitudes[i] > 0:
            direction = normals[i] / magnitudes[i]
            suppressed_normals[i] = normals[i] - suppression_factors[i] * direction

    # Compute new magnitudes after suppression
    suppressed_magnitudes = np.linalg.norm(suppressed_normals, axis=1)

    print(f"\nAfter suppression:")
    print(f"  Min: {suppressed_magnitudes.min():.4f}")
    print(f"  Max: {suppressed_magnitudes.max():.4f}")
    print(f"  Mean: {suppressed_magnitudes.mean():.4f}")

    # Find which control points had strong flow (now suppressed)
    strong_flow_threshold = np.percentile(magnitudes, 75)
    strong_flow_mask = magnitudes > strong_flow_threshold

    # Find which control points have relatively stronger signal after suppression
    # (These are the "revealed" dispersed signals)
    revealed_threshold = np.percentile(suppressed_magnitudes, 75)
    revealed_mask = suppressed_magnitudes > revealed_threshold

    print(f"\nStrong flow before suppression: {strong_flow_mask.sum()} control points")
    print(f"Strong signals after suppression: {revealed_mask.sum()} control points")

    # Check if different points are revealed
    newly_revealed = revealed_mask & ~strong_flow_mask
    print(f"Newly revealed (was weak, now strong): {newly_revealed.sum()} control points")

    return {
        'suppressed_normals': suppressed_normals,
        'suppressed_magnitudes': suppressed_magnitudes,
        'strong_flow_mask': strong_flow_mask,
        'revealed_mask': revealed_mask,
        'newly_revealed': newly_revealed
    }


def analyze_revealed_content(surface, chunks, newly_revealed_mask):
    """
    What content was revealed after flow suppression?
    """
    print(f"\n--- Content Analysis: Newly Revealed Chunks ---")

    revealed_chunks = []
    for idx, is_revealed in enumerate(newly_revealed_mask):
        if is_revealed:
            # Get chunk index
            i = idx // surface.grid_n
            j = idx % surface.grid_n
            msg_idx = surface.provenance['cp_to_msg'][(i, j)]
            revealed_chunks.append((msg_idx, chunks[msg_idx]))

    print(f"\nFound {len(revealed_chunks)} newly revealed chunks")

    if len(revealed_chunks) == 0:
        print("  (No chunks revealed - try different suppression parameters)")
        return

    # Extract characters from revealed chunks
    all_characters = Counter()
    for msg_idx, chunk in revealed_chunks:
        words = word_tokenize(chunk)
        pos_tags = pos_tag(words)
        proper_nouns = [word for word, tag in pos_tags if tag == 'NNP']
        all_characters.update(proper_nouns)

    print(f"\nTop characters in revealed chunks:")
    for char, count in all_characters.most_common(10):
        print(f"  {char:15s} {count:3d}")

    # Show a few examples
    print(f"\nExample revealed chunks:")
    for msg_idx, chunk in revealed_chunks[:3]:
        preview = chunk[:150].replace('\n', ' ')
        print(f"  Chunk {msg_idx}: {preview}...")


def main():
    print("="*70)
    print("EXPERIMENT: Flow Suppression to Reveal Dispersed Signals")
    print("="*70)

    # Load Frankenstein
    corpus_path = Path("benchmarks/initiative_proof_of_concept/corpus/frankenstein.txt")
    print(f"\nLoading {corpus_path}...")
    with open(corpus_path, 'r', encoding='utf-8') as f:
        text = f.read()

    # Chunk and build surface
    print("Chunking text...")
    chunks = chunk_text(text, chunk_size=1000)
    print(f"Created {len(chunks)} chunks")

    print(f"\nBuilding semantic surface...")
    surface = SemanticSurface(chunks)
    print(f"Surface grid: {surface.grid_m} x {surface.grid_n}")

    # Compute normals
    print(f"\nComputing surface normals...")
    normals, magnitudes = compute_surface_normals(surface)
    print(f"Computed {len(normals)} normal vectors")

    # Test all three suppression functions
    print(f"\n{'='*70}")
    print("TESTING ALL SUPPRESSION FUNCTIONS")
    print(f"{'='*70}")

    functions = [
        (lambda m: suppression_exponential(m, alpha=1.5), "Exponential (alpha=1.5)"),
        (lambda m: suppression_power(m, beta=2.5), "Power Law (beta=2.5)"),
        (lambda m: suppression_sigmoid(m, gamma=3.0), "Sigmoid (gamma=3.0)")
    ]

    results = {}
    for fn, name in functions:
        result = apply_flow_suppression(surface, normals, magnitudes, fn, name)
        results[name] = result

        # Analyze what was revealed
        analyze_revealed_content(surface, chunks, result['newly_revealed'])

    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}\n")

    print("Question: Does suppressing cohesive flow reveal dispersed signals?")
    print("\nResults:")
    for name, result in results.items():
        revealed_count = result['newly_revealed'].sum()
        print(f"  {name}: {revealed_count} chunks revealed")

    print("\nInterpretation:")
    print("  If this works, we should see:")
    print("    - Elizabeth/Justine appearing in revealed chunks")
    print("    - Content that was 'washed out' by Felix/cottage flow")
    print("    - Different narrative threads becoming visible")


if __name__ == "__main__":
    main()
