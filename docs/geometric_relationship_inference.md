# Geometric Relationship Inference

**Status:** Validated proof of concept
**Date:** November 2025

## Overview

Luna Nine's geometric relationship inference system detects semantic relationships between messages using **geometric properties alone** - no manual labeling, no LLM inference required.

**Core hypothesis:** Semantic relationships (necessary/opposed/ancillary/hierarchical/irrelevant) have distinct geometric signatures when messages are embedded on Bézier surfaces.

## Validation Results

Tested on 465 message pairs from 500 conversations:

- **95.48% variance explained** by geometric features
- **Silhouette score: 0.179** (clustering quality)
- **Multiple geometric signatures discovered** for relationship types
- Opposition detected with two distinct patterns:
  - **Type 1:** High curvature + low similarity (topic mismatch)
  - **Type 2:** Lower curvature + negative similarity (direct contradiction)

## How It Works

### 1. Geometric Feature Extraction

For each message pair, extract 33 geometric features:

**Distance metrics:**
- Geodesic distance (surface path length)
- Euclidean distance (embedding space)
- Cosine similarity

**Path properties:**
- Total path curvature
- Normalized curvature
- Arc length
- Tangent alignment

**Surface topology:**
- Gaussian curvature at each point
- Mean curvature at each point
- Surface normal alignment
- Influence overlap between regions

### 2. Dimensionality Reduction

- Normalize features (StandardScaler)
- Reduce to 2D using UMAP
- Preserve local and global structure

### 3. Clustering

- Group pairs by geometric similarity (k-means or DBSCAN)
- Each cluster represents a relationship type
- Validate using silhouette score and Davies-Bouldin index

### 4. Relationship Inference

Infer semantic relationships from cluster signatures:

| Relationship | Curvature | Similarity | Distance |
|-------------|-----------|------------|----------|
| **Opposed** | High | Low/Negative | Medium |
| **Necessary** | Low | High | Low |
| **Hierarchical** | Medium | High | Medium |
| **Ancillary** | Medium | Medium | Medium |
| **Irrelevant** | Any | Low | High |

## Usage

### Basic Pipeline

```python
from luna9.labeling import GeometricLabelingPipeline

# Initialize pipeline
pipeline = GeometricLabelingPipeline(
    parquet_dir='path/to/conversations',
    output_dir='./results',
    encoder_model='all-MiniLM-L6-v2'
)

# Run full analysis
results = pipeline.run_full_pipeline(
    sample_size=500,
    pair_strategy='mixed',  # sequential + distant + random
    max_pairs=1000,
    n_clusters=5,
    clustering_method='kmeans'
)
```

### Individual Components

```python
from luna9.labeling import (
    load_conversations,
    sample_pairs,
    GeometricFeatureExtractor,
    cluster_by_geometry
)

# Load data
messages = load_conversations('data.parquet')

# Sample message pairs
pairs = sample_pairs(messages, strategy='mixed', max_pairs=500)

# Extract geometric features
extractor = GeometricFeatureExtractor(surface, embeddings_dict)
features = extractor.extract_all_pairs(pairs)

# Cluster and infer relationships
clusters, labels = cluster_by_geometry(
    features,
    n_clusters=5,
    method='kmeans'
)
```

## Applications

### 1. Fast Semantic Search

Use geometric signatures to filter candidates before full LLM inference:

- Hash (u,v) coordinates into buckets → O(1) lookup
- Verify geometric properties → ~1ms
- Full LLM inference only for ambiguous cases → token savings

**Expected performance:** Handle 80% of queries geometrically, LLM for remaining 20%.

### 2. Prompt Injection Detection

Attacks create geometric opposition to system prompts. Detect before LLM sees input:

- **Single-shot detection:** Check if input opposes system prompt
- **Progressive unraveling:** Track drift across conversation turns
- **Context stuffing:** Detect surface expansion from irrelevant content

**Defense characteristics:**
- Fast (<100ms per check)
- Cheap (no token costs)
- Adaptive (learns from attacks geometrically)
- Evasion-resistant (geometry, not keywords)

### 3. Multi-Agent Coordination

Agents check geometric relationships between their plans:

- Detect conflicting objectives (opposed)
- Identify prerequisite steps (necessary)
- Find supporting context (ancillary)

### 4. Hallucination Detection

Compare generated responses to source material:

- High opposition → likely hallucination
- High necessity → well-grounded
- Low influence overlap → invention

### 5. Context Window Management

Use geometric properties for intelligent memory paging:

- High curvature → important transitions, keep in context
- Low curvature → redundant, safe to page out
- Opposition → conflicts to resolve, priority retention

## Architecture

### Two-Tier Inference System

**Tier 1 - Geometric (Fast & Cheap):**
- Hash-based lookup: O(1)
- Geometric verification: ~1ms
- Handles 80% of decisions

**Tier 2 - LLM (Slow & Expensive):**
- Full reasoning for ambiguous cases
- Only invoked when geometric confidence is low
- Learns from results to improve Tier 1

### Token Economy

Traditional approach:
```
Every semantic check → LLM call → 100-500 tokens
```

Geometric approach:
```
80% queries → geometric check → 0 tokens
20% queries → LLM call → 100-500 tokens
Average: 20-100 tokens per query (80-90% savings)
```

## Technical Details

### Feature Engineering

The 33 geometric features capture different aspects of semantic relationships:

1. **Distance features** (3): How far apart are the messages?
2. **Curvature features** (8): How complex is the semantic transition?
3. **Topology features** (10): What's the local surface structure?
4. **Alignment features** (6): Do messages point in similar directions?
5. **Influence features** (6): Do messages affect overlapping regions?

### Scaling Considerations

**Current implementation:**
- Suitable for 1K-10K message pairs
- UMAP + k-means clustering
- ~2-3 minutes for 500 conversations

**Production optimization needed:**
- Hash-based bucketing for O(1) lookup
- Incremental surface updates
- Distributed feature extraction
- GPU acceleration for embedding

## Future Work

### Short-Term
1. Implement hash bucketing system (`luna9.hash_index`)
2. Add geometric security daemon for prompt injection
3. Integrate with domain query system
4. Build two-tier inference API

### Medium-Term
1. Extend to other relationship types (causal, temporal, hierarchical)
2. Learn optimal feature weights from usage
3. Add confidence calibration
4. Deploy in production with real-time inference

### Long-Term
1. Multi-language support
2. Domain-specific feature engineering
3. Federated learning across installations
4. Self-improving geometric signatures

## References

### Inspiration

- **SolveSpace:** Bézier surface mathematics
- **Audfprint:** Hash bucketing for audio fingerprinting (adapted for semantic surfaces)

### Related Work

- Vector databases (pinecone, weaviate): Linear distance metrics only
- Graph neural networks: Require explicit graph structure
- LLM-based inference: Expensive, not provenance-tracked

**Geometric inference advantage:** Combines navigation efficiency with semantic richness, full provenance.

## Contributing

The geometric relationship inference system is under active development. Contributions welcome:

- Additional geometric features
- Alternative clustering methods
- Domain-specific adaptations
- Performance optimizations

See main README for contribution guidelines.
