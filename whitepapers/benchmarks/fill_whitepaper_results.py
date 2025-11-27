"""Fill in benchmark results into the whitepaper."""

import json
from pathlib import Path

# Load results
with open('whitepapers/benchmark_results_100.json') as f:
    results_100 = json.load(f)
with open('whitepapers/benchmark_results_500.json') as f:
    results_500 = json.load(f)
with open('whitepapers/benchmark_results_1000.json') as f:
    results_1000 = json.load(f)

# Read whitepaper
whitepaper_path = Path('whitepapers/01_parametric_surfaces.md')
content = whitepaper_path.read_text(encoding='utf-8')

# Fill in recall/precision table
recall_table = f"""| System | Recall@5 | Recall@10 | Recall@20 |
|--------|----------|-----------|-----------|
| Luna9 (100 chunks) | **{results_100['avg_recall_at_5']:.3f}** | **{results_100['avg_recall_at_10']:.3f}** | **{results_100['avg_recall_at_20']:.3f}** |
| Luna9 (500 chunks) | **{results_500['avg_recall_at_5']:.3f}** | **{results_500['avg_recall_at_10']:.3f}** | **{results_500['avg_recall_at_20']:.3f}** |
| Luna9 (1000 chunks) | **{results_1000['avg_recall_at_5']:.3f}** | **{results_1000['avg_recall_at_10']:.3f}** | **{results_1000['avg_recall_at_20']:.3f}** |"""

# Fill in latency table for 100 chunks
latency_100 = f"""| System | Mean | p50 | p95 | p99 |
|--------|------|-----|-----|-----|
| Luna9 (100 chunks) | **{results_100['mean_latency_ms']:.2f}** | **{results_100['p50_latency_ms']:.2f}** | **{results_100['p95_latency_ms']:.2f}** | **{results_100['p99_latency_ms']:.2f}** |"""

# Fill in latency table for 1000 chunks
latency_1000 = f"""| System | Mean | p50 | p95 | p99 |
|--------|------|-----|-----|-----|
| Luna9 (1000 chunks) | **{results_1000['mean_latency_ms']:.2f}** | **{results_1000['p50_latency_ms']:.2f}** | **{results_1000['p95_latency_ms']:.2f}** | **{results_1000['p99_latency_ms']:.2f}** |"""

# Fill in memory table
memory_table = f"""| System | Storage per message | Total (1000 chunks) |
|--------|---------------------|---------------------|
| Luna9 (100 chunks) | {results_100['memory_per_message_kb']:.2f} KB | {results_100['memory_mb']:.2f} MB |
| Luna9 (500 chunks) | {results_500['memory_per_message_kb']:.2f} KB | {results_500['memory_mb']:.2f} MB |
| Luna9 (1000 chunks) | **{results_1000['memory_per_message_kb']:.2f} KB** | **{results_1000['memory_mb']:.2f} MB** |"""

# Replace placeholders
replacements = {
    "**10K messages:**\n| System | Mean | p50 | p95 | p99 |\n|--------|------|-----|-----|-----|\n| ChromaDB (exact) | [X.X] | [X.X] | [X.X] | [X.X] |\n| ChromaDB (HNSW) | [X.X] | [X.X] | [X.X] | [X.X] |\n| Luna9 (no hash) | [X.X] | [X.X] | [X.X] | [X.X] |\n| Luna9 (with hash) | **[X.X]** | **[X.X]** | **[X.X]** | **[X.X]** |":
    f"**100 messages:**\n{latency_100}",

    "**100K messages:**\n| System | Mean | p50 | p95 | p99 |\n|--------|------|-----|-----|-----|\n| ChromaDB (exact) | [X.X] | [X.X] | [X.X] | [X.X] |\n| ChromaDB (HNSW) | [X.X] | [X.X] | [X.X] | [X.X] |\n| Luna9 (no hash) | [X.X] | [X.X] | [X.X] | [X.X] |\n| Luna9 (with hash) | **[X.X]** | **[X.X]** | **[X.X]** | **[X.X]** |":
    f"**1000 messages:**\n{latency_1000}",

    "| System | Recall@5 | Recall@10 | Recall@20 |\n|--------|----------|-----------|-----------|\n| ChromaDB (exact) | [0.XX] | [0.XX] | [0.XX] |\n| ChromaDB (HNSW) | [0.XX] | [0.XX] | [0.XX] |\n| Luna9 (smooth mode) | [0.XX] | [0.XX] | [0.XX] |\n| Luna9 (exact mode) | [0.XX] | [0.XX] | [0.XX] |\n| Luna9 (dual mode) | **[0.XX]** | **[0.XX]** | **[0.XX]** |":
    recall_table,

    "| System | Storage per message | 100K total |\n|--------|---------------------|------------|\n| ChromaDB | [X.XX] KB | [XX] MB |\n| Luna9 (surface only) | [X.XX] KB | [XX] MB |\n| Luna9 (surface + hash) | **[X.XX] KB** | **[XX] MB** |":
    memory_table
}

for old, new in replacements.items():
    content = content.replace(old, new)

# Add note about corpus and queries
corpus_note = f"""
**Corpus:** Pride and Prejudice by Jane Austen (Project Gutenberg)
- Chunked using Luna9's smart chunker (respects sentence boundaries)
- Chunk size: ~500 characters
- 100 chunks = ~50KB text
- 500 chunks = ~250KB text
- 1000 chunks = ~500KB text

**Query set:** {results_100['num_queries']} thematic queries
- "passages about marriage", "passages about love", etc.
- Ground truth: passages containing the search term
"""

# Insert after "### 4.1 Dataset"
content = content.replace(
    "### 4.1 Dataset\n\n**Corpus:** Project Gutenberg books (public domain literature)\n- 10K messages: ~[X] tokens, [Y] books\n- 50K messages: ~[X] tokens, [Y] books\n- 100K messages: ~[X] tokens, [Y] books\n\n**Query set:** [N] manually crafted queries spanning:\n- Factual retrieval (\"What color was the dress?\")\n- Thematic search (\"passages about betrayal\")\n- Character-focused (\"conversations between Alice and the Queen\")",
    f"### 4.1 Dataset\n{corpus_note}"
)

# Write back
whitepaper_path.write_text(content, encoding='utf-8')
print(f"[OK] Whitepaper updated with benchmark results!")
print(f"\nKey findings:")
print(f"  Recall@20: {results_100['avg_recall_at_20']:.1%} (100 chunks)")
print(f"  Query latency: {results_100['p50_latency_ms']:.0f}ms → {results_1000['p50_latency_ms']:.0f}ms (100 → 1000 chunks)")
print(f"  Memory: {results_1000['memory_per_message_kb']:.1f} KB/message")
