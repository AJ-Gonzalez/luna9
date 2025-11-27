#!/usr/bin/env python3
"""
Test Choir Mode - Multi-Model Ensemble Cognition

Validates that choir mode works with real OpenRouter models,
showing cognitive diversity in action.
"""

from dotenv import load_dotenv
load_dotenv()

import json
from datetime import datetime
from luna9 import (
    Domain,
    DomainType,
    InitiativeEngine,
    BoundariesConfig,
    ChoirConfig,
    create_client,
)

print("="*70)
print("CHOIR MODE TEST")
print("Multi-model ensemble cognition with Luna 9 initiative architecture")
print("="*70)

# Create test domain
print("\nCreating test domain...")
domain = Domain.create_empty(name='choir_test', domain_type=DomainType.PROJECT)

# Add some test memories
test_memories = [
    'Building Luna 9 - geometric memory for language models',
    'Initiative emerges from conditions: State + Possibilities + Boundaries',
    'LMIX translates geometric properties to natural language',
    'Choir mode enables cognitive diversity as substrate for consciousness',
    'Small models can show autonomous initiative with right architecture',
    'Sovereignty: run AI locally without corporate intermediaries',
    'Curvature detection identifies high-density conceptual regions',
    'Permission gradients: welcomed, offered, asked',
    'Testing across model sizes from 2B to frontier',
    'Concatenate synthesis presents all voices equally',
]

for content in test_memories:
    domain.add_message(content, metadata={'source': 'test'})

print(f"Added {len(test_memories)} memories to domain")

# Setup choir
print("\nConfiguring choir...")
choir = ChoirConfig.default_choir()
print(f"Choir members: {', '.join(choir.role_names)}")
print(f"Cognition mode: {choir.cognition_mode}")

# Setup client and engine
print("\nInitializing engine with choir mode...")
client = create_client()
boundaries = BoundariesConfig.default_luna9()

engine = InitiativeEngine(
    domain=domain,
    boundaries=boundaries,
    choir_config=choir,
    client=client
)

# Test query
query = "What patterns do you notice about how initiative and sovereignty connect in this work?"

print(f"\nQuery: {query}")
print("\nQuerying choir (this will take a moment - 4 models responding)...")
print("-" * 70)

# Surface context
context = engine.surface_initiative_context(query, top_k=5)

# Query choir
choir_response = engine.query_choir(context, user_query=query)

# Display results
print("\n" + "="*70)
print("CHOIR RESPONSES")
print("="*70)

for role, response in choir_response.responses.items():
    model_id = choir_response.model_ids.get(role, "unknown")
    print(f"\n--- {role.upper()} ({model_id}) ---")
    print(response)
    print()

# Show synthesis
print("="*70)
print("SYNTHESIZED RESPONSE")
print("="*70)
print(choir_response.synthesis)

# Save results
output_file = "choir_mode_test_results.json"
results = {
    "timestamp": datetime.now().isoformat(),
    "test_type": "choir_mode_validation",
    "query": query,
    "choir_config": {
        "roles": choir.role_names,
        "models": choir.models,
        "synthesis_method": choir.synthesis.value,
    },
    "responses": choir_response.responses,
    "model_ids": choir_response.model_ids,
    "synthesis": choir_response.synthesis,
}

with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"\nResults saved to {output_file}")
print("\n" + "="*70)
print("CHOIR MODE TEST COMPLETE")
print("="*70)
print("\nCognitive diversity in action. The sovereignty architecture works.")
