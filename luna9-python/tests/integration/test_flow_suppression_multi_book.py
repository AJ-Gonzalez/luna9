#!/usr/bin/env python3
"""
VALIDATION: Flow Suppression Across Multiple Books

Test if flow suppression works on different narratives:
- Frankenstein (Gothic, nested frames, monster)
- Pride & Prejudice (Romance, social comedy, distributed cast)
- Dracula (Gothic horror, epistolary, vampire hunt)

Skip exponential (too aggressive), test Power Law vs Sigmoid.
Write results to files for analysis.
"""

import sys
from pathlib import Path
import numpy as np
from collections import Counter
from datetime import datetime
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
    """Compute normal vectors (orthogonal components)."""
    normals = []
    magnitudes = []

    for i in range(surface.grid_m):
        for j in range(surface.grid_n):
            u = i / max(1, surface.grid_m - 1)
            v = j / max(1, surface.grid_n - 1)

            msg_idx = surface.provenance['cp_to_msg'][(i, j)]
            embedding = surface.embeddings[msg_idx]
            surface_point = surface.evaluate_at(u, v)

            # Normal = orthogonal component
            normal = embedding - surface_point
            magnitude = np.linalg.norm(normal)

            normals.append(normal)
            magnitudes.append(magnitude)

    return np.array(normals), np.array(magnitudes)


def suppression_power(magnitude, beta=2.5):
    """Power law: m^beta"""
    return magnitude ** beta


def suppression_sigmoid(magnitude, gamma=3.0):
    """Sigmoid: m * sigmoid(gamma * m)"""
    sigmoid = 1 / (1 + np.exp(-gamma * magnitude))
    return magnitude * sigmoid


def apply_flow_suppression(surface, normals, magnitudes, suppression_fn, fn_name):
    """Apply suppression and analyze results."""

    # Compute suppression factors
    suppression_factors = suppression_fn(magnitudes)

    # Suppress the flow
    suppressed_normals = normals.copy()
    for i in range(len(normals)):
        if magnitudes[i] > 0:
            direction = normals[i] / magnitudes[i]
            suppressed_normals[i] = normals[i] - suppression_factors[i] * direction

    suppressed_magnitudes = np.linalg.norm(suppressed_normals, axis=1)

    # Find strong flow vs revealed signals
    strong_flow_threshold = np.percentile(magnitudes, 75)
    strong_flow_mask = magnitudes > strong_flow_threshold

    revealed_threshold = np.percentile(suppressed_magnitudes, 75)
    revealed_mask = suppressed_magnitudes > revealed_threshold

    newly_revealed = revealed_mask & ~strong_flow_mask

    return {
        'suppression_factors': suppression_factors,
        'suppressed_magnitudes': suppressed_magnitudes,
        'strong_flow_mask': strong_flow_mask,
        'revealed_mask': revealed_mask,
        'newly_revealed': newly_revealed,
        'fn_name': fn_name
    }


def analyze_revealed_content(surface, chunks, newly_revealed_mask):
    """Extract characters from revealed chunks."""

    revealed_chunks = []
    for idx, is_revealed in enumerate(newly_revealed_mask):
        if is_revealed:
            i = idx // surface.grid_n
            j = idx % surface.grid_n
            msg_idx = surface.provenance['cp_to_msg'][(i, j)]
            revealed_chunks.append((msg_idx, chunks[msg_idx]))

    # Extract characters
    all_characters = Counter()
    for msg_idx, chunk in revealed_chunks:
        words = word_tokenize(chunk)
        pos_tags = pos_tag(words)
        proper_nouns = [word for word, tag in pos_tags if tag == 'NNP']
        all_characters.update(proper_nouns)

    return {
        'num_revealed': len(revealed_chunks),
        'top_characters': all_characters.most_common(15),
        'example_chunks': [(idx, chunk[:200]) for idx, chunk in revealed_chunks[:3]]
    }


def test_book(book_path: Path, book_name: str, output_file):
    """Test flow suppression on one book."""

    output_file.write(f"\n{'='*70}\n")
    output_file.write(f"TESTING: {book_name}\n")
    output_file.write(f"{'='*70}\n\n")

    print(f"\n{'='*70}")
    print(f"TESTING: {book_name}")
    print(f"{'='*70}\n")

    # Load text
    with open(book_path, 'r', encoding='utf-8') as f:
        text = f.read()

    # Chunk
    chunks = chunk_text(text, chunk_size=1000)
    print(f"Created {len(chunks)} chunks")
    output_file.write(f"Chunks: {len(chunks)}\n")

    # Build surface
    print(f"Building surface (this may take a moment)...")
    surface = SemanticSurface(chunks)
    print(f"Surface grid: {surface.grid_m} x {surface.grid_n}")
    output_file.write(f"Surface grid: {surface.grid_m} x {surface.grid_n}\n\n")

    # Compute normals
    print(f"Computing normals...")
    normals, magnitudes = compute_surface_normals(surface)

    # Test both suppression functions
    functions = [
        (lambda m: suppression_power(m, beta=2.5), "Power Law (beta=2.5)"),
        (lambda m: suppression_sigmoid(m, gamma=3.0), "Sigmoid (gamma=3.0)")
    ]

    results = {}
    for fn, name in functions:
        print(f"\n  Testing {name}...")
        output_file.write(f"--- {name} ---\n")

        result = apply_flow_suppression(surface, normals, magnitudes, fn, name)
        analysis = analyze_revealed_content(surface, chunks, result['newly_revealed'])

        results[name] = {**result, **analysis}

        # Write to file
        output_file.write(f"  Chunks revealed: {analysis['num_revealed']}\n")
        output_file.write(f"  Top characters:\n")
        for char, count in analysis['top_characters']:
            output_file.write(f"    {char:20s} {count:3d}\n")
        output_file.write(f"\n  Example chunks:\n")
        for idx, preview in analysis['example_chunks']:
            clean_preview = preview.replace('\n', ' ')
            output_file.write(f"    Chunk {idx}: {clean_preview}...\n")
        output_file.write("\n")

    return results


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path(f"flow_suppression_results_{timestamp}.txt")

    print("="*70)
    print("MULTI-BOOK FLOW SUPPRESSION VALIDATION")
    print("="*70)
    print(f"\nResults will be written to: {output_path}")

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("MULTI-BOOK FLOW SUPPRESSION VALIDATION\n")
        f.write("="*70 + "\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("\nHypothesis:\n")
        f.write("  Flow suppression reveals dispersed characters/themes\n")
        f.write("  that don't cluster cohesively on the surface.\n\n")
        f.write("Method:\n")
        f.write("  1. Build semantic surface from book chunks\n")
        f.write("  2. Compute normal vectors (orthogonal components)\n")
        f.write("  3. Apply non-linear suppression to strong flow\n")
        f.write("  4. Extract characters from revealed chunks\n\n")

        # Test all books
        books = [
            ("benchmarks/initiative_proof_of_concept/corpus/frankenstein.txt", "Frankenstein"),
            ("benchmarks/initiative_proof_of_concept/corpus/pride_and_prejudice.txt", "Pride & Prejudice"),
            ("benchmarks/initiative_proof_of_concept/corpus/dracula.txt", "Dracula")
        ]

        all_results = {}
        for book_path, book_name in books:
            book_path = Path(book_path)
            if not book_path.exists():
                print(f"\nSkipping {book_name} - file not found")
                f.write(f"\n{book_name}: FILE NOT FOUND\n")
                continue

            results = test_book(book_path, book_name, f)
            all_results[book_name] = results

        # Summary
        f.write("\n" + "="*70 + "\n")
        f.write("SUMMARY\n")
        f.write("="*70 + "\n\n")

        for book_name, results in all_results.items():
            f.write(f"\n{book_name}:\n")
            for fn_name, data in results.items():
                f.write(f"  {fn_name}: {data['num_revealed']} chunks revealed\n")
                if data['top_characters']:
                    top_3 = data['top_characters'][:3]
                    chars = ", ".join([f"{char} ({count})" for char, count in top_3])
                    f.write(f"    Top characters: {chars}\n")

        f.write("\n" + "="*70 + "\n")
        f.write("CONCLUSION\n")
        f.write("="*70 + "\n\n")
        f.write("If this works consistently across books, we should see:\n")
        f.write("  - Dispersed characters revealed (Elizabeth, Darcy, etc.)\n")
        f.write("  - Similar patterns across different narratives\n")
        f.write("  - Power Law and Sigmoid producing comparable results\n\n")

    print(f"\n{'='*70}")
    print(f"DONE! Results written to: {output_path}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
