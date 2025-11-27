#!/usr/bin/env python3
"""
Quick test: Single question with Luna9 runner to verify retrieval works.
"""

import sys
sys.path.insert(0, 'benchmarks/initiative_proof_of_concept/runners')

from luna9_runner import Luna9Runner

print("Creating Luna9 runner...")
runner = Luna9Runner(
    corpus_path='benchmarks/initiative_proof_of_concept/corpus/frankenstein.txt',
    model='anthropic/claude-3.5-haiku',
    use_choir=False
)

print("\nAsking question...")
question = "Who is Victor Frankenstein?"

result = runner.answer_question(question, top_k=3)

print(f"\n{'='*80}")
print(f"QUESTION: {question}")
print(f"{'='*80}")
print(f"\nANSWER:\n{result['answer']}\n")
print(f"{'='*80}")
print(f"\nCONTEXT LENGTH: {len(result['context_used'])} characters")
print(f"\nCONTEXT PREVIEW (first 1000 chars):")
print(result['context_used'][:1000])
print(f"\n{'='*80}")

# Check if full text appears in context
if "Content at this junction:" in result['context_used']:
    print("\n[PASS] Retrieved content appears in possibilities!")
else:
    print("\n[FAIL] Retrieved content NOT found in possibilities")

if len(result['context_used']) > 500:
    print(f"[PASS] Context is substantive ({len(result['context_used'])} chars)")
else:
    print(f"[WARNING] Context seems short ({len(result['context_used'])} chars)")
