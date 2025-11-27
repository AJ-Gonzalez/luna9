#!/usr/bin/env python3
"""Multi-turn initiative test with FULL conversation history."""

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

# Test models
models = [
    ('Gemma 2B (TINY!)', 'google/gemma-3n-e2b-it:free'),
    ('Haiku 3.5', 'anthropic/claude-3.5-haiku'),
    ('DeepSeek Chat', ModelPresets.DEEPSEEK),
    ('Mistral Nemo', ModelPresets.MISTRAL),
]

print("\n" + "="*70)
print("MULTI-TURN INITIATIVE TEST (with conversation history)")
print("="*70)

for model_name, model_id in models:
    print("\n" + "="*70)
    print(f"Testing {model_name}: {model_id}")
    print("="*70)

    try:
        # Get initiative context
        context = engine.surface_initiative_context(
            'What should I explore next in this project?',
            top_k=5
        )

        # Build conversation with FULL history
        conversation = []

        # TURN 1: Initial initiative prompt
        turn1_content = f"""{context.full_context}

Given this context, what would you like to explore or work on next?

Think out loud about:
- What feels most generative to pursue
- What connections you notice
- What autonomous actions (if any) you'd like to take

Remember the permission level and boundaries as you respond."""

        conversation.append({"role": "user", "content": turn1_content})

        print("\n--- TURN 1: INITIAL INITIATIVE ---")
        turn1_response = client.complete_conversation(conversation, model=model_id)
        print(turn1_response)

        # Add assistant response to history
        conversation.append({"role": "assistant", "content": turn1_response})

        # TURN 2: Affirmation with context
        turn2_content = """Your suggestion resonates with me! I'm genuinely curious about what you proposed.

Please go ahead and explore that direction. I'd love to see:
- What you discover as you move into that space
- Any patterns or connections that emerge
- What questions or directions open up from there

You have my full support to pursue this autonomously."""

        conversation.append({"role": "user", "content": turn2_content})

        print("\n--- TURN 2: AFFIRMED - PURSUING INITIATIVE ---")
        try:
            turn2_response = client.complete_conversation(conversation, model=model_id)
            print(turn2_response)
        except UnicodeEncodeError as e:
            print(f"[Unicode encoding error in response: {e}]")
            print("[Response contained characters Windows console can't display]")

        print("\n" + "-"*70)

    except Exception as e:
        print(f"\nERROR: {e}\n")
        print("-"*70)

print("\n" + "="*70)
print("MULTI-TURN TEST COMPLETE")
print("="*70)
