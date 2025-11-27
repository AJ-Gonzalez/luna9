#!/usr/bin/env python3
"""
Test: Does our surface capture narrative structure?

If the geometric surface preserves semantic meaning, then:
- Chunks about Felix should cluster together in (u,v) space
- Chunks about Justine should cluster together
- Chunks about different narrative sections should be geometrically separated

Let's find out.
"""

import sys
from pathlib import Path
import json
import numpy as np
from collections import Counter, defaultdict
import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag

# Add luna9 to path
sys.path.insert(0, str(Path(__file__).parent))

from luna9.core.semantic_surface import SemanticSurface


def extract_narrative_markers(text: str) -> dict:
    """
    Extract narrative markers from a text chunk.

    Returns:
        Dict with key characters, themes, linguistic features
    """
    # Tokenize and POS tag
    words = word_tokenize(text)
    pos_tags = pos_tag(words)

    # Extract proper nouns (named entities)
    proper_nouns = [word for word, tag in pos_tags if tag == 'NNP']
    entity_counts = Counter(proper_nouns)

    # Key characters to track
    key_characters = {
        'Victor': entity_counts.get('Victor', 0),
        'Elizabeth': entity_counts.get('Elizabeth', 0),
        'Justine': entity_counts.get('Justine', 0),
        'Clerval': entity_counts.get('Clerval', 0),
        'Felix': entity_counts.get('Felix', 0),
        'Safie': entity_counts.get('Safie', 0),
        'Agatha': entity_counts.get('Agatha', 0),
        'Walton': entity_counts.get('Walton', 0),
    }

    # Key themes (word presence)
    words_lower = [w.lower() for w in words if w.isalnum()]
    theme_markers = {
        'creature': words_lower.count('creature') + words_lower.count('monster'),
        'death': words_lower.count('death') + words_lower.count('died'),
        'science': words_lower.count('science') + words_lower.count('experiment'),
        'cottage': words_lower.count('cottage'),
        'letter': words_lower.count('letter'),
        'murder': words_lower.count('murder') + words_lower.count('murdered'),
    }

    return {
        'characters': key_characters,
        'themes': theme_markers,
        'total_entities': len(proper_nouns),
        'chunk_preview': text[:100].replace('\n', ' ')
    }


def chunk_text(text: str, chunk_size: int = 1000) -> list:
    """Same chunking as benchmark runner."""
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


def analyze_surface_narrative_structure():
    """
    Main experiment: Do narrative sections cluster on the surface?
    """
    print("="*70)
    print("EXPERIMENT: Narrative Structure on Semantic Surface")
    print("="*70)

    # Load Frankenstein
    corpus_path = Path("benchmarks/initiative_proof_of_concept/corpus/frankenstein.txt")
    print(f"\nLoading {corpus_path}...")
    with open(corpus_path, 'r', encoding='utf-8') as f:
        text = f.read()

    # Chunk it (same as benchmark)
    print("Chunking text...")
    chunks = chunk_text(text, chunk_size=1000)
    print(f"Created {len(chunks)} chunks")

    # Build surface
    print(f"\nBuilding semantic surface with MPNet embeddings...")
    surface = SemanticSurface(chunks)
    print(f"Surface grid: {surface.grid_m} x {surface.grid_n} = {surface.grid_m * surface.grid_n} control points")

    # Extract narrative markers for each chunk
    print(f"\nExtracting narrative markers from each chunk...")
    chunk_markers = []
    for i, chunk in enumerate(chunks):
        if i % 50 == 0:
            print(f"  Processing chunk {i}/{len(chunks)}...")
        markers = extract_narrative_markers(chunk)
        chunk_markers.append(markers)

    # Map chunks to surface coordinates
    print(f"\nMapping chunks to surface coordinates...")
    chunk_positions = []
    for i in range(len(chunks)):
        # Get control point position
        cp_i, cp_j = surface.provenance['msg_to_cp'][i]
        u = cp_i / max(1, surface.grid_m - 1)
        v = cp_j / max(1, surface.grid_n - 1)
        chunk_positions.append((u, v))

    print(f"\n{'='*70}")
    print("NARRATIVE CLUSTERING ANALYSIS")
    print(f"{'='*70}\n")

    # Analyze clustering for each character
    characters = ['Victor', 'Elizabeth', 'Justine', 'Clerval', 'Felix', 'Safie', 'Agatha', 'Walton']

    for character in characters:
        # Find chunks that mention this character significantly
        character_chunks = []
        for i, markers in enumerate(chunk_markers):
            count = markers['characters'][character]
            if count >= 3:  # At least 3 mentions
                character_chunks.append((i, count, chunk_positions[i]))

        if not character_chunks:
            continue

        print(f"\n--- {character} ---")
        print(f"Appears in {len(character_chunks)} chunks")

        # Calculate geometric spread
        positions = np.array([pos for _, _, pos in character_chunks])
        if len(positions) > 1:
            # Calculate centroid
            centroid = positions.mean(axis=0)
            # Calculate average distance from centroid
            distances = np.sqrt(((positions - centroid) ** 2).sum(axis=1))
            avg_distance = distances.mean()

            print(f"Centroid: ({centroid[0]:.3f}, {centroid[1]:.3f})")
            print(f"Avg distance from centroid: {avg_distance:.3f}")
            print(f"Spread: {'TIGHT' if avg_distance < 0.15 else 'MODERATE' if avg_distance < 0.3 else 'DISPERSED'}")

            # Show some example chunks
            print(f"\nExample chunk positions:")
            for chunk_id, count, (u, v) in character_chunks[:5]:
                print(f"  Chunk {chunk_id:3d} at ({u:.3f}, {v:.3f}) - {count} mentions")

    # Analyze theme clustering
    print(f"\n{'='*70}")
    print("THEME CLUSTERING ANALYSIS")
    print(f"{'='*70}\n")

    themes = ['creature', 'cottage', 'letter', 'murder', 'science']

    for theme in themes:
        # Find chunks with this theme
        theme_chunks = []
        for i, markers in enumerate(chunk_markers):
            count = markers['themes'][theme]
            if count >= 2:  # At least 2 mentions
                theme_chunks.append((i, count, chunk_positions[i]))

        if not theme_chunks:
            continue

        print(f"\n--- {theme.upper()} ---")
        print(f"Appears in {len(theme_chunks)} chunks")

        positions = np.array([pos for _, _, pos in theme_chunks])
        if len(positions) > 1:
            centroid = positions.mean(axis=0)
            distances = np.sqrt(((positions - centroid) ** 2).sum(axis=1))
            avg_distance = distances.mean()

            print(f"Centroid: ({centroid[0]:.3f}, {centroid[1]:.3f})")
            print(f"Avg distance from centroid: {avg_distance:.3f}")
            print(f"Spread: {'TIGHT' if avg_distance < 0.15 else 'MODERATE' if avg_distance < 0.3 else 'DISPERSED'}")

    # Final analysis: Do chunks with similar content cluster?
    print(f"\n{'='*70}")
    print("HYPOTHESIS TEST")
    print(f"{'='*70}\n")

    print("Question: Does our surface preserve narrative structure?")
    print("\nIf YES: Characters/themes should have TIGHT clustering")
    print("If NO:  Characters/themes should be DISPERSED randomly")
    print("\nResults above show the actual clustering patterns.")
    print("\nInterpretation:")
    print("  TIGHT     < 0.15 = Strong geometric clustering")
    print("  MODERATE  < 0.30 = Some clustering, some spread")
    print("  DISPERSED > 0.30 = Random/no geometric structure")


if __name__ == "__main__":
    analyze_surface_narrative_structure()
