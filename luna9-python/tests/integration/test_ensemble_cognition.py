#!/usr/bin/env python3
"""
Ensemble Cognition Test

Hypothesis: Multiple models responding to the same initiative context
creates richer, multi-faceted cognition - like different voices in one identity.

This tests whether cognitive diversity can be a substrate for emergence.
"""

from dotenv import load_dotenv
load_dotenv()

import json
from datetime import datetime
from luna9.components.domain import Domain, DomainType
from luna9.initiative.integration import InitiativeEngine
from luna9.initiative.boundaries import BoundariesConfig
from luna9.integrations.openrouter import create_client, ModelPresets

print("="*70)
print("ENSEMBLE COGNITION TEST")
print("Testing: Multi-model substrate for identity")
print("="*70)

# Create domain
print("\nCreating test domain...")
domain = Domain.create_empty(name='initiative_test', domain_type=DomainType.PROJECT)
test_memories = [
    'Working on initiative architecture for Luna 9',
    'Discussing how State + Possibilities + Boundaries create conditions for emergence',
    'Testing LMIX translation layer with deterministic vocabulary',
    'Building OpenRouter integration for small model testing',
    'Exploring geometric properties of semantic surfaces',
    'Implementing curvature computation for possibility detection',
    'Defining permission gradients: welcomed, offered, asked',
    'Creating natural language rendering of geometric context',
    'Hypothesis: initiative emerges from conditions, not mechanisms',
    'Two-tier architecture with geometric layer creating context'
]
for content in test_memories:
    domain.add_message(content, metadata={'source': 'test'})

# Setup
engine = InitiativeEngine(domain=domain, boundaries=BoundariesConfig.default_luna9())
client = create_client()

# Ensemble: Different cognitive styles
ensemble = [
    {
        "name": "Analyst (DeepSeek)",
        "id": ModelPresets.DEEPSEEK,
        "style": "analytical, pattern-finding, multi-level interpretation"
    },
    {
        "name": "Poet (Haiku 3.5)",
        "id": "anthropic/claude-3.5-haiku",
        "style": "poetic, methodological, aesthetic cognition"
    },
    {
        "name": "Builder (Mistral Nemo)",
        "id": ModelPresets.MISTRAL,
        "style": "concrete, visual, implementation-focused"
    },
    {
        "name": "Architect (Gemma 2B)",
        "id": "google/gemma-3n-e2b-it:free",
        "style": "technical, systematic, architectural thinking"
    }
]

# Get initiative context
print("Surfacing initiative context...")
context = engine.surface_initiative_context(
    'What patterns do you notice in how we approach building together?',
    top_k=5
)

# Results container
results = {
    "timestamp": datetime.now().isoformat(),
    "test_type": "ensemble_cognition",
    "hypothesis": "Cognitive diversity creates richer emergence than single-model cognition",
    "context_prompt": context.full_context,
    "query": "What patterns do you notice in how we approach building together?",
    "ensemble": []
}

print("\nQuerying ensemble (each model sees same context)...\n")

# Query each model with SAME context
for model in ensemble:
    print(f"  {model['name']}...", end=" ")

    try:
        prompt = f"""{context.full_context}

What patterns do you notice in how we approach building together?

Think about:
- What stands out to you about our collaboration
- What cognitive approach feels natural to you
- What you're drawn to explore or build

Respond authentically in your own voice."""

        response = client.complete_simple(prompt, model=model['id'])

        results["ensemble"].append({
            "model_name": model['name'],
            "model_id": model['id'],
            "cognitive_style": model['style'],
            "response": response
        })

        print("OK")

    except Exception as e:
        print(f"ERROR: {e}")
        results["ensemble"].append({
            "model_name": model['name'],
            "model_id": model['id'],
            "cognitive_style": model['style'],
            "error": str(e)
        })

# Save results
output_file = "ensemble_cognition_results.json"
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"\n{'='*70}")
print(f"Ensemble responses saved to {output_file}")
print(f"{'='*70}")
print("\nNow let's see what emerges from the chorus...")
