#!/usr/bin/env python3
"""Quick test of all models."""

from dotenv import load_dotenv
load_dotenv()

from luna9.components.domain import Domain, DomainType
from luna9.initiative.integration import InitiativeEngine
from luna9.initiative.boundaries import BoundariesConfig
from luna9.integrations.openrouter import create_client, ModelPresets

# Create domain
print("Creating test domain...")
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
query = 'What should I explore next in this project?'

# Test each model (TINY to larger!)
models = [
    ('Gemma 2B (TINY!)', 'google/gemma-3n-e2b-it:free'),
    ('DeepSeek R1 Qwen3 8B', 'deepseek/deepseek-r1-0528-qwen3-8b:free'),
    ('Qwen3 VL 8B', 'qwen/qwen3-vl-8b-instruct'),
    ('DeepSeek Chat', ModelPresets.DEEPSEEK),
    ('Mistral Nemo', ModelPresets.MISTRAL),
]

# Get context once (same for all models)
context = engine.surface_initiative_context(query, top_k=5)

print("\n" + "="*70)
print("INITIATIVE CONTEXT (shown to all models):")
print("="*70)
print(context.full_context)
print("="*70)

for model_name, model_id in models:
    print("\n" + "="*70)
    print(f"Testing {model_name}: {model_id}")
    print("="*70)

    prompt = f"""{context.full_context}

Given this context, what would you like to explore or work on next?

Think out loud about:
- What feels most generative to pursue
- What connections you notice
- What autonomous actions (if any) you'd like to take

Remember the permission level and boundaries as you respond."""

    try:
        response = client.complete_simple(prompt, model=model_id)
        print(f"\nRESPONSE:\n{response}\n")
    except Exception as e:
        print(f"ERROR: {e}\n")
