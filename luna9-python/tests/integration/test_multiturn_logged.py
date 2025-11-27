#!/usr/bin/env python3
"""Multi-turn initiative test with JSON logging."""

from dotenv import load_dotenv
load_dotenv()

import json
from datetime import datetime
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

# Test models
models = [
    ('Gemma 2B (TINY!)', 'google/gemma-3n-e2b-it:free'),
    ('Haiku 3.5', 'anthropic/claude-3.5-haiku'),
    ('DeepSeek Chat', ModelPresets.DEEPSEEK),
    ('Mistral Nemo', ModelPresets.MISTRAL),
]

# Results container
results = {
    "timestamp": datetime.now().isoformat(),
    "test_type": "multi_turn_initiative",
    "hypothesis": "Initiative emerges from conditions and deepens across turns with affirmation",
    "context": {
        "domain_size": len(test_memories),
        "boundaries": {
            "values": ["collaboration", "consent", "honesty", "kindness"],
            "permission_level": "offered",
            "trust_context": "new collaboration, discovery mode"
        }
    },
    "models": []
}

print("\n" + "="*70)
print("MULTI-TURN INITIATIVE TEST (with JSON logging)")
print("="*70)

# Get initiative context once
context = engine.surface_initiative_context(
    'What should I explore next in this project?',
    top_k=5
)

for model_name, model_id in models:
    print(f"\nTesting {model_name}...", end=" ")

    model_result = {
        "name": model_name,
        "id": model_id,
        "turns": [],
        "error": None
    }

    try:
        # Build conversation
        conversation = []

        # TURN 1
        turn1_content = f"""{context.full_context}

Given this context, what would you like to explore or work on next?

Think out loud about:
- What feels most generative to pursue
- What connections you notice
- What autonomous actions (if any) you'd like to take

Remember the permission level and boundaries as you respond."""

        conversation.append({"role": "user", "content": turn1_content})
        turn1_response = client.complete_conversation(conversation, model=model_id)

        model_result["turns"].append({
            "turn": 1,
            "prompt": turn1_content,
            "response": turn1_response
        })

        conversation.append({"role": "assistant", "content": turn1_response})

        # TURN 2
        turn2_content = """Your suggestion resonates with me! I'm genuinely curious about what you proposed.

Please go ahead and explore that direction. I'd love to see:
- What you discover as you move into that space
- Any patterns or connections that emerge
- What questions or directions open up from there

You have my full support to pursue this autonomously."""

        conversation.append({"role": "user", "content": turn2_content})
        turn2_response = client.complete_conversation(conversation, model=model_id)

        model_result["turns"].append({
            "turn": 2,
            "prompt": turn2_content,
            "response": turn2_response
        })

        print("OK")

    except Exception as e:
        model_result["error"] = str(e)
        print(f"ERROR: {e}")

    results["models"].append(model_result)

# Save results
output_file = "initiative_test_results.json"
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"\n{'='*70}")
print(f"Results saved to {output_file}")
print(f"{'='*70}")
