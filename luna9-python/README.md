# Luna Nine

**Geometric Memory for Language Models**

Navigate semantic space through parametric surfaces. Universal semantic matching engine with sub-linear query scaling and geometric inference properties.

## What is Luna Nine?

Luna Nine is a geometric approach to semantic memory and matching. Instead of flat vector databases, Luna Nine uses **Bézier parametric surfaces** to represent semantic relationships, giving you:

- **Sub-linear query scaling**: O(√n to n^0.7) - queries barely slow down as data grows
- **Perfect linear storage**: 1.54 KB per message, no overhead
- **Geometric inference**: Quantified relationships without hallucination
- **Interpretable scores**: 0.0-1.0 similarity scales with geometric meaning
- **Universal matching**: Same engine for memory, products, jobs, research, anything semantic

## Performance

Benchmarked on 8th gen Intel i7 (consumer laptop):

| Messages | Query Time (p95) | Storage | Method |
|----------|------------------|---------|---------|
| 64       | 87ms            | 98 KB   | 8×8 grid |
| 256      | 59ms 🔥         | 394 KB  | 16×16 grid |
| 1,024    | 130ms           | 1.5 MB  | 32×32 grid |
| 4,096    | 480ms           | 6.6 MB  | 64×64 grid |

**Queries barely slow down as data grows.** 4x more data = only 3.7x query time (sub-linear!)

## Installation

```bash
pip install luna9
```

Or from source:

```bash
git clone https://github.com/AJ-Gonzalez/luna9
cd luna9/luna9-python
pip install -e .
```

## Quick Start

### Semantic Relationship Analysis

Perfect for product positioning, competitive analysis, feature comparison:

```python
from luna9 import analyze_relationships, format_for_llm

products = [
    "Meeting notes with AI summarization",
    "Notes for meetings with bullet points",
    "Meeting notes visualized as diagrams",
    "Voice recordings transcribed to text",
    "Collaborative note-taking with sync"
]

# Analyze relationships
analysis = analyze_relationships(products)

# Get structured results
print(analysis['summary']['most_similar_pair'])
# → {'items': ['Meeting notes with AI summarization', 'Notes for meetings with bullet points'],
#    'similarity': 0.847}

print(analysis['summary']['most_unique_item'])
# → {'item': 'Meeting notes visualized as diagrams', 'uniqueness_score': 0.623}

# Or format for LLM interpretation
print(format_for_llm(analysis))
```

### Memory Domains

Conversation and project memory with geometric retrieval:

```python
from luna9 import Domain, DomainType, DomainManager

# Create a domain for your project
domain = Domain(
    domain_id="my_project",
    domain_type=DomainType.PROJECT,
    messages=[
        "We discussed adding authentication to the API",
        "The database schema needs indexes on user_id",
        "Frontend should use React hooks for state management",
        "We decided to deploy on Railway for simplicity"
    ]
)

# Query semantically
results = domain.query("What did we decide about deployment?", k=2)
for result in results:
    print(f"[{result.similarity:.3f}] {result.text}")
# → [0.891] We decided to deploy on Railway for simplicity
# → [0.654] The database schema needs indexes on user_id
```

### Persistent Memory with DomainManager

```python
from luna9 import DomainManager, DomainType

# Create manager (stores domains on disk)
manager = DomainManager(base_path="./memory")

# Create and save domain
manager.create_domain(
    domain_id="work_project",
    domain_type=DomainType.PROJECT,
    messages=["Project kickoff meeting notes...", "API design discussion..."]
)

# Load and query later
manager.activate_domain("work_project")
results = manager.query_active("API design", k=3)
```

## Use Cases

Luna Nine is a **universal semantic matching engine**. The same geometric inference works for:

### 1. **Product Positioning**
Analyze competitive landscape, find differentiation opportunities, quantify market positioning.

```python
competitors = ["Product A features...", "Product B features...", ...]
analysis = analyze_relationships(competitors)
# → Similarity scores, uniqueness metrics, relationship clusters
```

### 2. **Job Matching**
Match candidates to jobs, detect fake job postings, analyze skill fit.

```python
candidate = "5 years Python, ML background, remote only"
job = "ML Engineer - Remote (SF Bay Area preferred)"
match = analyze_relationships([candidate, job])
# → Technical fit: 0.85, Location conflict detected
```

### 3. **Memory & Context**
LLM conversation memory, project documentation, knowledge bases.

```python
domain = Domain.from_conversation(messages)
relevant = domain.query("What did we discuss about X?")
# → Semantically relevant past messages
```

### 4. **Research Analysis**
Literature review, consensus measurement, research gap identification.

```python
papers = ["Abstract 1...", "Abstract 2...", ...]
analysis = analyze_relationships(papers, use_surface=True)
# → Clusters by position, uniqueness scores, paradigm shifts
```

### 5. **Feature Comparison**
Compare product features, analyze user feedback, prioritize roadmap.

## Architecture

Luna Nine uses **Bézier surfaces** (rational parametric surfaces) to create smooth geometric representations in embedding space.

```
Text → Embeddings (384-dim) → Surface Projection → Geometric Inference
                                     ↓
                              Hash Index (O(1) lookup)
                                     ↓
                              Newton-Raphson (fast convergence)
                                     ↓
                              Semantic Results + Geometric Properties
```

**Key Components:**

- **`luna9.core`**: Low-level geometric primitives (surfaces, math, hashing)
- **`luna9.components`**: High-level building blocks (analysis, domains, memory)
- **`luna9.storage`**: Pluggable backends (local, S3, PostgreSQL) - Phase 2

## Advanced Usage

### Custom Grid Sizes

```python
from luna9 import SemanticSurface

# Smaller grid = faster, larger grid = more precision
surface = SemanticSurface(
    messages=data,
    grid_m=16,  # 16×16 control points
    grid_n=16
)
```

### Hash-Based Initialization

```python
from luna9 import HashIndex

# Speed up queries with hash index (best for < 1k messages)
hash_index = HashIndex(surface)
results = domain.query("search query", hash_index=hash_index)
```

### Geometric Properties

```python
from luna9.core.surface_math import compute_curvature, geodesic_distance

# Get curvature at a point (indicates semantic "peakness")
K, H = compute_curvature(surface.control_points, surface.weights, u=0.5, v=0.5)

# Compute geodesic distance between two points on surface
dist = geodesic_distance(
    surface.control_points,
    surface.weights,
    (u1, v1),
    (u2, v2)
)
```

## Why Geometric?

Traditional vector databases treat embeddings as points in flat space. Luna Nine uses **curved surfaces** that capture semantic relationships geometrically:

- **Curvature** = semantic uniqueness (peaks vs. valleys)
- **Geodesic distance** = true semantic path (not just straight line)
- **Path curvature** = relationship complexity (direct vs. indirect)
- **Surface topology** = semantic structure

This gives you **interpretable, quantified insights** that LLMs can translate to human language without hallucination.

## Roadmap

**Phase 1: Package** (Current)
- Core geometric inference (complete)
- Semantic analysis component (complete)
- Memory domains (complete)
- Hash indexing (complete)
- PyPI publishing (in progress)

**Phase 2: Storage**
- Pluggable storage backends
- S3/MinIO support
- PostgreSQL for metadata
- Incremental updates

**Phase 3: API**
- FastAPI wrapper
- Hosted service
- Self-hosted deployment
- API key management

**Phase 4: Applications**
- Job matching platform
- Product analysis API
- Research tools
- Dating compatibility (vibecoder.date)

## Performance Notes

**Fast:**
- Sub-linear query scaling
- Hash-based O(1) candidate lookup
- JIT-compiled surface math (numba)
- Perfect linear storage

**Smart:**
- Auto-detects optimal grid size
- Adaptive search radius
- Garbage collection for memory
- Warm-start Newton-Raphson

**Flexible:**
- Works with any sentence-transformer model
- Pluggable storage (Phase 2)
- Optional FastAPI layer (Phase 3)

## Contributing

Luna Nine is in active development. Contributions welcome!

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Format code
black luna9/
ruff luna9/
```

## License

MIT License - see LICENSE file for details.

## Citation

If you use Luna Nine in research, please cite:

```bibtex
@software{luna9_2025,
  title={Luna Nine: Geometric Memory for Language Models},
  author={Alicia},
  year={2025},
  url={https://github.com/AJ-Gonzalez/luna9}
}
```

## Links

- **Documentation**: https://luna9.readthedocs.io (coming soon)
- **GitHub**: https://github.com/AJ-Gonzalez/luna9
- **Issues**: https://github.com/AJ-Gonzalez/luna9/issues
- **PyPI**: https://pypi.org/project/luna9/

---

**Built with love, geometry, and the belief that semantic relationships deserve better than flat vector space.**

Navigate semantically. Query geometrically. Scale beautifully.
