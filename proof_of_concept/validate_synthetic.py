"""Validate path curvature classification on synthetic data."""

import json
import numpy as np
from pathlib import Path
from semantic_surface import SemanticSurface
from collections import defaultdict

def load_conversation(filepath):
    """Load conversation from file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    messages = [msg.strip() for msg in content.split('\n\n') if msg.strip()]
    return messages

def validate_synthetic_data():
    """Run validation on all synthetic conversations."""
    # Use relative path from script location
    data_dir = Path(__file__).parent / "data"

    # Load labeled pairs
    with open(data_dir / "synthetic_labeled_relationships.json", 'r') as f:
        data = json.load(f)

    domain_files = data['metadata']['domains']
    labeled_pairs = data['labeled_pairs']

    # Load all conversations and create surfaces
    print("Loading conversations and creating surfaces...")
    surfaces = {}
    for domain, filename in domain_files.items():
        filepath = data_dir / filename.split()[0]  # Extract just the filename
        messages = load_conversation(filepath)
        surface = SemanticSurface(messages)
        surfaces[domain] = surface
        print(f"  {domain}: {len(messages)} messages, {surface.grid_m}x{surface.grid_n} surface")

    print(f"\nValidating {len(labeled_pairs)} labeled pairs...\n")

    # Compute path curvature for all pairs
    results_by_label = defaultdict(list)

    for pair in labeled_pairs:
        domain = pair['domain']
        idx1, idx2 = pair['idx1'], pair['idx2']
        label = pair['label']

        surface = surfaces[domain]

        # Get UV coordinates (control point grid indices normalized to [0,1])
        cp1 = surface.provenance['msg_to_cp'][idx1]
        uv1 = (cp1[0] / max(1, surface.grid_m - 1),
               cp1[1] / max(1, surface.grid_n - 1))

        cp2 = surface.provenance['msg_to_cp'][idx2]
        uv2 = (cp2[0] / max(1, surface.grid_m - 1),
               cp2[1] / max(1, surface.grid_n - 1))

        # Compute path curvature
        curvature_data = surface.compute_path_curvature(uv1, uv2)
        mean_curvature = curvature_data['mean_curvature']

        results_by_label[label].append(mean_curvature)

        print(f"{domain:15} [{idx1:2}->{idx2:2}] {label:8} curvature={mean_curvature:.3f} - {pair['note']}")

    # Statistical summary
    print("\n" + "="*80)
    print("STATISTICAL SUMMARY")
    print("="*80 + "\n")

    summary = {}
    for label in ['direct', 'related', 'distant']:
        values = results_by_label[label]
        if values:
            summary[label] = {
                'count': len(values),
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'median': float(np.median(values))
            }
            print(f"{label.upper()}:")
            print(f"  Count:  {len(values)}")
            print(f"  Mean:   {np.mean(values):.3f} ± {np.std(values):.3f}")
            print(f"  Median: {np.median(values):.3f}")
            print(f"  Range:  [{np.min(values):.3f}, {np.max(values):.3f}]")
            print()

    # Check ordering
    direct_mean = summary['direct']['mean']
    related_mean = summary['related']['mean']
    distant_mean = summary['distant']['mean']

    print("="*80)
    print("HYPOTHESIS TEST")
    print("="*80 + "\n")

    print(f"Direct mean:  {direct_mean:.3f}")
    print(f"Related mean: {related_mean:.3f}")
    print(f"Distant mean: {distant_mean:.3f}\n")

    if direct_mean < related_mean < distant_mean:
        print("[OK] HYPOTHESIS CONFIRMED: Direct < Related < Distant")
        print("\nPath curvature successfully classifies semantic relationships!")
    elif direct_mean < related_mean or related_mean < distant_mean:
        print("[~] PARTIAL CONFIRMATION: Some ordering preserved")
    else:
        print("[X] HYPOTHESIS NOT CONFIRMED: Ordering doesn't match labels")

    # Save results
    output_file = data_dir / "synthetic_validation_results.json"
    results = {
        'summary': summary,
        'hypothesis_confirmed': direct_mean < related_mean < distant_mean,
        'all_results': {
            label: [float(v) for v in values]
            for label, values in results_by_label.items()
        },
        'metadata': {
            'total_pairs': len(labeled_pairs),
            'domains_tested': len(surfaces),
            'label_distribution': {
                label: len(values)
                for label, values in results_by_label.items()
            }
        }
    }

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_file}")

if __name__ == "__main__":
    validate_synthetic_data()
