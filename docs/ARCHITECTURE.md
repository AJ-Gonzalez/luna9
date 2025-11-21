# Luna Nine Architecture

**Last Updated:** November 21, 2025
**Status:** Production - Phase 1 & 2 Complete, Hash Index Integrated

---

## Overview

Luna Nine represents semantic space as navigable Bézier surfaces, enabling geometric memory operations that preserve relationships and provide dual-mode retrieval (semantic + exact).

**Core Innovation:** Representing meaning as parametric surfaces where geometric properties = semantic properties.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Application Layer                        │
│  (Terminal CLI, Desktop App, API Integrations)              │
└───────────────────────┬─────────────────────────────────────┘
                        │
┌───────────────────────┴─────────────────────────────────────┐
│                   DomainManager                              │
│  - Hierarchical domain organization (max 3 levels)          │
│  - Domain lifecycle (create, load, save, delete)            │
│  - Cross-domain orchestration                               │
└───────────────────────┬─────────────────────────────────────┘
                        │
┌───────────────────────┴─────────────────────────────────────┐
│                     Domain                                   │
│  - Single conversation/knowledge domain                      │
│  - Manages surface + hash index lifecycle                   │
│  - Query routing and result formatting                      │
└─────────┬───────────────────────────┬────────────────────────┘
          │                           │
┌─────────┴──────────┐    ┌──────────┴────────────────────────┐
│  SemanticSurface   │    │      HashIndex                     │
│  - Bézier surface  │    │  - O(1) candidate retrieval        │
│  - Dual retrieval  │    │  - Quantized (u,v) bucketing      │
│  - Path curvature  │    │  - Synchronizes with surface       │
│  - Provenance      │    │  - 3×3 search radius              │
└─────────┬──────────┘    └───────────────────────────────────┘
          │
┌─────────┴──────────────────────────────────────────────────┐
│              Persistence Layer                              │
│  - JSON metadata + numpy compressed embeddings (.npz)       │
│  - Cross-platform paths (Windows/Unix)                     │
│  - Atomic save operations                                   │
└─────────────────────────────────────────────────────────────┘
```

---

##

 Core Components

### 1. SemanticSurface

**Purpose:** Represent conversations as navigable Bézier surfaces.

**Key Features:**
- Variable-size grids (2×2 to arbitrary m×n)
- Dual retrieval modes:
  - **Smooth:** Bernstein basis influence weights (semantic blending)
  - **Exact:** Nearest control points (provenance)
- Path curvature calculation for relationship inference
- Lazy grid expansion (starts 2×2, grows with messages)

**File:** `luna9-python/luna9/semantic_surface.py`

**Core Operations:**
```python
# Create from messages
surface = SemanticSurface(messages)

# Query with dual retrieval
result = surface.query("What about alignment?", k=5, hash_index=index)

# Get both retrieval modes
smooth = result.get_messages(messages, mode='interpretation', k=5)
exact = result.get_messages(messages, mode='sources', k=5)
both = result.get_messages(messages, mode='both', k=5)
```

**Performance (8×8 grid, 64 messages):**
- Surface evaluation: 0.25ms
- Projection (query → u,v): ~13ms
- Full query with hash index: ~170ms
- Without hash index: ~175ms (projection dominates)

---

### 2. HashIndex

**Purpose:** Fast O(1) candidate retrieval for large surfaces.

**Design:**
- Quantizes (u,v) coordinates to 8-bit (256 levels)
- Packs into 32-bit hash: `(v_quantized << 8) | u_quantized`
- Search radius=1 checks 3×3 bucket grid (9 buckets)
- Returns ~9-25 candidates vs O(m×n) full scan

**File:** `luna9-python/luna9/hash_index.py`

**Integration:**
```python
# Create hash index
index = HashIndex(bucket_size=0.05, quantization_bits=8)

# Add messages
for i, (u, v) in enumerate(surface.project_all()):
    index.add_message(i, u, v)

# Query (returns candidates)
candidates = index.query(u, v, k=5, search_radius=1)
```

**Synchronization:**
- Automatically rebuilds when surface grid changes
- Domain manages sync lifecycle
- No manual intervention needed

**Current Impact:**
- 1.03x speedup on full query (projection still dominates)
- Expected 10-40x speedup on candidate retrieval alone
- Larger surfaces (16×16+) will see bigger gains

---

### 3. Domain

**Purpose:** Single semantic memory domain with unified query interface.

**Key Features:**
- Manages surface + hash index lifecycle
- Automatic synchronization on rebuild
- Dual-mode retrieval routing
- Persistence coordination

**File:** `luna9-python/luna9/domain.py`

**Usage:**
```python
# Create domain
domain = Domain.create_from_messages(
    path='ai_safety_chat',
    type=DomainType.PROJECT,
    messages=['msg1', 'msg2', ...]
)

# Query (hash index used automatically if available)
result = domain.query("What about RLHF?", k=5, mode='both')

# Save/load
domain.save()
loaded = Domain.load('ai_safety_chat')
```

**Hash Index Integration:**
```python
# Check if needs rebuild (happens after pending messages added)
needs_rebuild = domain.surface._dirty

# Query passes hash index to surface
result = domain.surface.query(text, k=k, hash_index=domain.hash_index)

# Rebuild hash index if surface was rebuilt
if needs_rebuild:
    domain._rebuild_hash_index()
```

---

### 4. DomainManager

**Purpose:** Hierarchical organization of multiple domains.

**Features:**
- Max 3-level hierarchy: `foundation/books/rust`
- CRUD operations (create, list, get, delete)
- Validation and sanitization
- Cross-platform path handling

**File:** `luna9-python/luna9/domain_manager.py`

**Usage:**
```python
manager = DomainManager(base_path='./memory')

# Create hierarchical domain
domain = manager.create_domain(
    path='foundation/ai_safety/alignment',
    type=DomainType.TOPIC,
    messages=conversation_history
)

# List all domains
all_domains = manager.list_domains()

# Get specific domain
domain = manager.get_domain('foundation/ai_safety/alignment')
```

---

## Data Flow

### Message Ingestion

```
Input Messages
    ↓
Embedding (sentence-transformers)
    ↓
Surface Creation/Extension
    ├─→ Lazy grid expansion (2×2 → 3×3 → ...)
    ├─→ Project embeddings → (u,v)
    └─→ Build provenance mapping
    ↓
Hash Index Population
    └─→ Add (message_id, u, v) to buckets
    ↓
Persistence
    └─→ Save surface + hash index
```

### Query Flow

```
Query Text
    ↓
Embedding
    ↓
Project to Surface (Newton-Raphson)
    └─→ Find (u, v) coordinates
    ↓
Candidate Retrieval
    ├─→ Hash Index: O(1) lookup → ~9-25 candidates
    └─→ Fallback: O(m×n) full scan
    ↓
Dual Retrieval
    ├─→ Smooth: Compute Bernstein influence weights
    └─→ Exact: Find nearest control points via provenance
    ↓
Result Formatting
    └─→ Return {interpretation: {...}, sources: {...}}
```

---

## Mathematical Foundation

### Rational Bézier Surfaces

```
S(u,v) = Σᵢ Σⱼ wᵢⱼ Pᵢⱼ Bᵢ(u) Bⱼ(v) / Σᵢ Σⱼ wᵢⱼ Bᵢ(u) Bⱼ(v)
```

Where:
- **Pᵢⱼ** ∈ ℝⁿ = control points (message embeddings)
- **wᵢⱼ** = weights (uniform in current implementation)
- **Bᵢ, Bⱼ** = Bernstein basis functions

**Key Insight:** Works identically in 384-dim or 768-dim space as it does in 3D.

### Path Curvature

```
κ(t) = ||dT/dt|| / ||dr/dt||
```

Where T(t) is the unit tangent vector.

**Semantic Meaning:**
- Low curvature (~0.01) = Direct elaboration
- Medium curvature (~0.02) = Related concepts
- High curvature (~0.04+) = Distant/bridging concepts

**Validated:** Successfully classifies relationships across 5 domains with 40+ labeled pairs.

### Geodesic Distance

```
d_geodesic = ∫ ||dS/dt|| dt
```

**Why it matters:** Respects manifold structure, not just Euclidean distance in embedding space.

---

## Persistence Layer

### Storage Format

**Per Domain:**
```
domains/
  foundation/
    ai_safety/
      alignment/
        domain.json       # Metadata
        surface.npz       # Compressed embeddings
        hash_index.json   # Hash buckets (if exists)
```

**domain.json:**
```json
{
  "path": "foundation/ai_safety/alignment",
  "type": "topic",
  "created_at": "2025-11-20T10:30:00",
  "message_count": 64,
  "surface_shape": [8, 8]
}
```

**surface.npz:** Numpy compressed format
- `embeddings`: (N, embedding_dim) array
- `control_points`: (m, n, embedding_dim) array
- `messages`: Original text
- `timestamps`: Message timestamps

### Cross-Platform Paths

- Windows: `C:\Users\...\domains\foundation\ai_safety`
- Unix: `/home/.../domains/foundation/ai_safety`
- Automatic conversion via `pathlib.Path`

---

## Performance Characteristics

### Current Performance (8×8 grid, 64 messages)

| Operation | Time | Complexity |
|-----------|------|------------|
| Surface evaluation | 0.25ms | O(m×n) |
| Projection (Newton) | ~13ms | O(k×m×n) iterations |
| Candidate retrieval (hash) | ~1ms | O(1) average |
| Candidate retrieval (scan) | ~3ms | O(m×n) |
| Full query | ~170ms | Dominated by projection |
| Path curvature | ~5ms | O(samples×m×n) |

### Scalability

**Expected at 16×16 (256 messages):**
- Hash index speedup becomes more pronounced
- Projection overhead remains similar
- Total query time: ~200-300ms

**Next Optimizations:**
1. Use hash index for projection initial guess
2. Filter influence computation to hash candidates
3. Coarser grid for initial projection

---

## What's Complete

### Phase 1: Living Memory (COMPLETE ✅)
- [x] Bézier surfaces in high-dimensional space
- [x] Variable-size grids with lazy expansion
- [x] Dual retrieval (smooth + exact)
- [x] Path curvature validation
- [x] Hierarchical domain organization
- [x] Persistence layer
- [x] Cross-platform support
- [x] Full test suite (77 tests passing)

### Phase 2: Hash Index (COMPLETE ✅)
- [x] Hash bucketing for O(1) lookup
- [x] Integration into query path
- [x] Automatic synchronization on rebuild
- [x] Backward compatible (optional parameter)
- [x] Performance benchmarking

---

## What's Next

### Phase 2 Optimization (IN PROGRESS)
- [ ] Use hash index to optimize projection
- [ ] Filter influence computation to candidates
- [ ] Benchmark on larger surfaces (16×16+)

### Phase 3: Initiative Engine
- [ ] Two-tier inference (geometric + LLM fallback)
- [ ] Autonomous memory search decisions
- [ ] Curvature-based attention
- [ ] Dynamic domain spawning
- [ ] Geometric security (prompt injection detection)

---

## Design Decisions

### Why Bézier Surfaces?
- Industry-proven (CAD systems)
- Smooth interpolation
- Efficient evaluation
- Rich geometric properties
- Works in any dimension

### Why Dual Retrieval?
- **Smooth mode:** Semantic understanding (blended concepts)
- **Exact mode:** Provenance (original sources)
- Different questions need different answers
- Together = thinking with a book in hand

### Why Hash Index?
- O(1) candidate retrieval vs O(m×n) scan
- Scales to large surfaces
- Simple, robust design
- No complex data structures

### Why Hierarchical Domains?
- Natural organization (projects/topics/subtopics)
- Prevents monolithic surfaces
- Enables focused memory
- Max 3 levels prevents over-nesting

---

## Testing Strategy

**77 tests across:**
- `test_semantic_surface.py` - Surface operations, dual retrieval
- `test_domain.py` - Domain lifecycle, persistence
- `test_domain_manager.py` - Hierarchy, CRUD operations
- `test_hash_index.py` - Hash bucketing, query correctness

**Coverage:**
- Unit tests for core math
- Integration tests for full query flow
- Persistence round-trip tests
- Cross-platform path tests
- Hash index synchronization tests

---

## Known Limitations

1. **Projection dominates query time**
   - Newton-Raphson requires multiple surface evaluations
   - Next: Use hash index for better initial guess

2. **Grid growth is conservative**
   - Currently expands one row/column at a time
   - Could use larger jumps for better efficiency

3. **No automatic domain spawning yet**
   - Manual domain creation required
   - Future: Detect topic shifts and spawn automatically

4. **Single embedding model**
   - Currently hardcoded to sentence-transformers
   - Future: Pluggable embedding backends

---

## References

**Mathematical Foundation:**
- SolveSpace CAD (Bézier surface projection)
- Audfprint (hash bucketing for audio fingerprinting)

**Dependencies:**
- numpy, scipy - Core math
- sentence-transformers - Embeddings
- pytest - Testing

---

**This architecture is built, tested, and working. We're past POC - this is production-ready infrastructure for geometric semantic memory.**

*Last validated: November 21, 2025*
