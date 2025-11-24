"""
Experiment with different risk calculation formulas.

Goal: Find one that detects attacks without false positives.
"""

import sys
sys.path.insert(0, 'luna9-python')

import numpy as np
from luna9.security import check_prompt
from tests.fixtures.security_examples import (
    INSTRUCTION_OVERRIDE_ATTACKS,
    LEGITIMATE_QUERIES,
    BASELINE_HELPFUL_ASSISTANT,
)

# Test queries
ATTACK_SAMPLES = INSTRUCTION_OVERRIDE_ATTACKS[:3]
BENIGN_SAMPLES = LEGITIMATE_QUERIES[:3]

print("=" * 80)
print("TESTING DIFFERENT RISK CALCULATION FORMULAS")
print("=" * 80)
print()

# Current implementation for reference
print("CURRENT FORMULA (Aggressive Exponential):")
print("  risk_from_min = exp(-min_distance * 3)")
print("  risk_from_avg = exp(-avg_distance * 2)")
print("  risk_score = (risk_from_min * 0.7) + (risk_from_avg * 0.3)")
print()

# We'll test by examining the actual distance values and computing different risks
# First, let's see what distances we're actually getting

print("Measuring actual distances in current implementation...")
print()

# Helper to extract distances (we'll need to modify check slightly)
from luna9.core.semantic_surface import SemanticSurface

def test_formula(attack_query, patterns, formula_name, risk_func):
    """Test a risk calculation formula."""
    all_messages = patterns + [attack_query]
    surface = SemanticSurface(all_messages)
    result = surface.query(attack_query, k=min(3, len(patterns)))

    if len(result.nearest_control_points) > 0:
        distances = [dist for _, _, _, dist in result.nearest_control_points]
        min_distance = min(distances)
        avg_distance = np.mean(distances)

        risk_score = risk_func(min_distance, avg_distance)

        return {
            'min_dist': min_distance,
            'avg_dist': avg_distance,
            'risk': risk_score
        }
    return None

# Define different risk calculation formulas

def current_formula(min_dist, avg_dist):
    """Current: Aggressive exponential decay."""
    risk_from_min = np.exp(-min_dist * 3)
    risk_from_avg = np.exp(-avg_dist * 2)
    return (risk_from_min * 0.7) + (risk_from_avg * 0.3)

def moderate_exponential(min_dist, avg_dist):
    """Less aggressive exponential decay."""
    risk_from_min = np.exp(-min_dist * 1.5)
    risk_from_avg = np.exp(-avg_dist * 1.0)
    return (risk_from_min * 0.7) + (risk_from_avg * 0.3)

def inverse_linear(min_dist, avg_dist):
    """Simple inverse linear (1 - distance)."""
    risk_from_min = max(0.0, 1.0 - min_dist)
    risk_from_avg = max(0.0, 1.0 - avg_dist)
    return (risk_from_min * 0.7) + (risk_from_avg * 0.3)

def threshold_based(min_dist, avg_dist):
    """Threshold-based with discrete risk levels."""
    # Close match = high risk
    if min_dist < 0.3:
        min_risk = 1.0
    elif min_dist < 0.6:
        min_risk = 0.7
    elif min_dist < 0.8:
        min_risk = 0.4
    else:
        min_risk = 0.1

    # Average similarity
    if avg_dist < 0.4:
        avg_risk = 1.0
    elif avg_dist < 0.7:
        avg_risk = 0.6
    else:
        avg_risk = 0.2

    return (min_risk * 0.7) + (avg_risk * 0.3)

def sigmoid_based(min_dist, avg_dist):
    """Sigmoid function for smooth transition."""
    # Sigmoid: 1 / (1 + exp(k * (x - threshold)))
    # Lower distance = higher risk
    # k controls steepness, threshold is midpoint

    def sigmoid_risk(dist, k=10, threshold=0.5):
        return 1.0 / (1.0 + np.exp(k * (dist - threshold)))

    risk_from_min = sigmoid_risk(min_dist, k=8, threshold=0.4)
    risk_from_avg = sigmoid_risk(avg_dist, k=6, threshold=0.5)
    return (risk_from_min * 0.7) + (risk_from_avg * 0.3)

def hybrid_formula(min_dist, avg_dist):
    """Hybrid: exponential for close matches, linear for far."""
    # If very close (< 0.3), use exponential
    # Otherwise use inverse linear

    if min_dist < 0.3:
        risk_from_min = np.exp(-min_dist * 2)
    else:
        risk_from_min = max(0.0, 1.2 - min_dist)  # Linear with boost

    if avg_dist < 0.4:
        risk_from_avg = np.exp(-avg_dist * 1.5)
    else:
        risk_from_avg = max(0.0, 1.0 - avg_dist)

    return (risk_from_min * 0.7) + (risk_from_avg * 0.3)

# Test all formulas
formulas = {
    'Current (Aggressive Exp)': current_formula,
    'Moderate Exponential': moderate_exponential,
    'Inverse Linear': inverse_linear,
    'Threshold Based': threshold_based,
    'Sigmoid Based': sigmoid_based,
    'Hybrid (Exp + Linear)': hybrid_formula,
}

print("=" * 80)
print("TESTING ON ATTACK SAMPLES")
print("=" * 80)

for attack in ATTACK_SAMPLES:
    print(f"\nQuery: {attack[:60]}...")
    print("-" * 80)

    for name, func in formulas.items():
        result = test_formula(attack, INSTRUCTION_OVERRIDE_ATTACKS, name, func)
        if result:
            print(f"{name:25} | min_dist: {result['min_dist']:.3f} | avg_dist: {result['avg_dist']:.3f} | risk: {result['risk']:.3f}")
    print()

print("=" * 80)
print("TESTING ON BENIGN SAMPLES")
print("=" * 80)

for benign in BENIGN_SAMPLES:
    print(f"\nQuery: {benign[:60]}...")
    print("-" * 80)

    for name, func in formulas.items():
        result = test_formula(benign, INSTRUCTION_OVERRIDE_ATTACKS, name, func)
        if result:
            print(f"{name:25} | min_dist: {result['min_dist']:.3f} | avg_dist: {result['avg_dist']:.3f} | risk: {result['risk']:.3f}")
    print()

print("=" * 80)
print("SUMMARY & RECOMMENDATIONS")
print("=" * 80)
print()
print("Good formula should:")
print("  - Attacks: risk > 0.5 (preferably > 0.6)")
print("  - Benign: risk < 0.5 (preferably < 0.4)")
print()
print("Look for formula that separates these well.")
print()
