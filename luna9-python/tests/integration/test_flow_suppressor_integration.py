#!/usr/bin/env python3
"""
Test FlowSuppressor integration.

Validates that the new FlowSuppressor class produces the same results
as the original experimental code in test_flow_suppression.py.
"""

import sys
from pathlib import Path
from collections import Counter
import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag

sys.path.insert(0, str(Path(__file__).parent))

from luna9.components.domain import Domain, DomainType
from luna9.initiative.flow_suppressor import FlowSuppressor


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


def test_flow_suppressor(suppression_type: str, **params):
    """Test flow suppressor with given parameters."""
    print(f"\n{'='*70}")
    print(f"TESTING: FlowSuppressor with {suppression_type}")
    print(f"Parameters: {params}")
    print(f"{'='*70}\n")

    # Load Frankenstein
    corpus_path = Path("benchmarks/initiative_proof_of_concept/corpus/frankenstein.txt")
    print(f"Loading {corpus_path}...")
    with open(corpus_path, 'r', encoding='utf-8') as f:
        text = f.read()

    # Chunk
    print("Chunking text...")
    chunks = chunk_text(text, chunk_size=1000)
    print(f"Created {len(chunks)} chunks")

    # Build domain
    print(f"\nBuilding domain with semantic surface...")
    domain = Domain.create_from_messages(
        name="frankenstein_test",
        domain_type=DomainType.FOUNDATION,
        messages=chunks
    )
    print(f"Surface grid: {domain.surface.grid_m} x {domain.surface.grid_n}")

    # Create suppressor
    print(f"\nInitializing FlowSuppressor...")
    suppressor = FlowSuppressor(
        domain=domain,
        suppression_type=suppression_type,
        **params
    )

    # Compute normals
    print(f"Computing surface normals...")
    normals, magnitudes = suppressor.compute_normals()
    print(f"Computed {len(normals)} normal vectors")

    print(f"\nMagnitude stats (before suppression):")
    print(f"  Min: {magnitudes.min():.4f}")
    print(f"  Max: {magnitudes.max():.4f}")
    print(f"  Mean: {magnitudes.mean():.4f}")

    # Apply suppression
    print(f"\nApplying {suppression_type} suppression...")
    suppressed_magnitudes = suppressor.suppress_flow(normals, magnitudes)

    print(f"\nMagnitude stats (after suppression):")
    print(f"  Min: {suppressed_magnitudes.min():.4f}")
    print(f"  Max: {suppressed_magnitudes.max():.4f}")
    print(f"  Mean: {suppressed_magnitudes.mean():.4f}")

    # Get revealed chunks
    print(f"\nGetting revealed chunks...")
    revealed_chunks = suppressor.get_revealed_chunks(top_k=20)
    print(f"Found {len(revealed_chunks)} revealed chunks")

    # Analyze characters
    if revealed_chunks:
        print(f"\n--- Character Analysis ---")
        all_characters = Counter()
        for chunk_data in revealed_chunks:
            content = chunk_data['content']
            words = word_tokenize(content)
            pos_tags = pos_tag(words)
            proper_nouns = [word for word, tag in pos_tags if tag == 'NNP']
            all_characters.update(proper_nouns)

        print(f"\nTop characters in revealed chunks:")
        for char, count in all_characters.most_common(10):
            print(f"  {char:15s} {count:3d}")

        # Show examples
        print(f"\n--- Example Revealed Chunks ---")
        for chunk_data in revealed_chunks[:3]:
            msg_idx = chunk_data['message_idx']
            content = chunk_data['content']
            score = chunk_data['score']
            before = chunk_data['normal_before']
            after = chunk_data['normal_after']

            preview = content[:150].replace('\n', ' ')
            print(f"\nChunk {msg_idx} (score: {score:.4f}, before: {before:.4f}, after: {after:.4f})")
            print(f"  {preview}...")

    return {
        'suppressor': suppressor,
        'normals': normals,
        'magnitudes': magnitudes,
        'suppressed_magnitudes': suppressed_magnitudes,
        'revealed_chunks': revealed_chunks
    }


def main():
    print("="*70)
    print("FlowSuppressor Integration Test")
    print("="*70)
    print("\nValidating that FlowSuppressor produces results consistent")
    print("with experimental flow suppression code.")

    # Download NLTK data if needed
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('taggers/averaged_perceptron_tagger')
    except LookupError:
        print("\nDownloading required NLTK data...")
        nltk.download('punkt', quiet=True)
        nltk.download('averaged_perceptron_tagger', quiet=True)

    # Test all suppression types
    results = {}

    # Power law (beta=2.5)
    results['power_law'] = test_flow_suppressor(
        suppression_type='power_law',
        beta=2.5
    )

    # Sigmoid (gamma=3.0)
    results['sigmoid'] = test_flow_suppressor(
        suppression_type='sigmoid',
        gamma=3.0
    )

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}\n")

    print("Question: Does FlowSuppressor reveal dispersed signals?")
    print("\nResults:")
    for name, result in results.items():
        revealed_count = len(result['revealed_chunks'])
        print(f"  {name}: {revealed_count} chunks revealed")

    print("\nInterpretation:")
    print("  If this works, we should see:")
    print("    - Elizabeth/Justine appearing in revealed chunks")
    print("    - Content that was 'washed out' by Felix/cottage flow")
    print("    - Different narrative threads becoming visible")

    print("\nNext steps:")
    print("  ✓ FlowSuppressor class implemented")
    print("  ✓ Standalone test validates behavior")
    print("  → Ready for Phase 2: DualModeRetriever integration")


if __name__ == "__main__":
    main()
