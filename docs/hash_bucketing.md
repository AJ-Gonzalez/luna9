# Hash Bucketing for Semantic Surfaces

**Status:** Design Phase
**Target:** Phase 2A Implementation
**Expected Speedup:** 40x for typical queries

---

## Overview

### The Problem

Semantic surface queries currently scale O(N) - every query must compute geometric features against all stored messages. For a domain with 1,000 messages, that's 1,000 distance calculations per query.

As domains grow to 10K, 100K, or millions of messages, this becomes untenable.

### The Solution

**Hash bucketing** based on surface coordinates. Messages with similar geometric positions hash to the same buckets, reducing query search space from O(N) to O(1) bucket lookup + O(bucket_size) verification.

**Key insight:** Since we already project messages onto 2D semantic surfaces with (u,v) coordinates, we can hash those coordinates directly. The surface has already done the hard work of dimensionality reduction and semantic organization.

---

## Inspiration: Audio Fingerprinting

This approach is adapted from `audfprint`, an audio fingerprinting system that identifies songs from noisy recordings.

### How audfprint Works

Audio fingerprinting identifies spectral peaks (landmarks) and packs their relationships into compact hashes:

```
Bits 0-5:   dt (time difference between peaks)
Bits 6-11:  df (frequency difference)
Bits 12-19: f1 (first peak frequency)
```

Songs with similar temporal/spectral patterns hash to the same buckets. Matching only checks bucket candidates, not the entire song database.

**Critical properties:**
- Multiple hashes per song (20/second)
- Fixed hash space (~1M buckets)
- Fixed bucket size (~100 entries)
- Graceful degradation (random drops from full buckets don't break matching)
- Temporal consistency verification for final matches

### Adaptation to Semantic Surfaces

Instead of audio landmarks, we hash **semantic surface coordinates**:

```
Bits 0-7:   du (u coordinate difference, quantized)
Bits 8-15:  dv (v coordinate difference, quantized)
Bits 16-23: u1 (first message u coordinate, quantized)
Bits 24-31: v1 (first message v coordinate, quantized)
```

Messages with similar geometric relationships hash to the same buckets.

---

## Algorithm Design

### Surface Coordinate Hashing

**Input:** Message projected to semantic surface → (u, v) coordinates ∈ [0, 1] × [0, 1]

**Quantization:**
```python
# 8 bits per coordinate = 256 discrete positions
u_quantized = int(u * 255)  # 0-255
v_quantized = int(v * 255)  # 0-255
```

**Hash Construction:**
```python
# 32-bit hash from two coordinate pairs
def compute_hash(u1, v1, u2, v2):
    du = int(abs(u2 - u1) * 255)  # Bits 0-7
    dv = int(abs(v2 - v1) * 255)  # Bits 8-15
    u1_q = int(u1 * 255)          # Bits 16-23
    v1_q = int(v1 * 255)          # Bits 24-31

    return (v1_q << 24) | (u1_q << 16) | (dv << 8) | du
```

**Hash space:** 2^32 ≈ 4.3 billion possible hashes (in practice, distribution will be denser in semantically active regions)

### Bucket Management

**Bucket capacity:** Default 100 messages per bucket (configurable)

**Collision strategy:**
- Buckets are lists of message references
- When bucket exceeds capacity, remove oldest entries (FIFO)
- Matching requires minimum N consistent hits (default 3), so losing popular entries doesn't prevent retrieval

**Storage structure:**
```python
hash_index = {
    hash_value: [
        (message_id, u, v, timestamp),
        (message_id, u, v, timestamp),
        ...  # up to 100 entries
    ]
}
```

### Query Process

```python
# 1. Project query to surface
query_point = surface.project(query_embedding)
u_q, v_q = query_point

# 2. Generate candidate hashes (check nearby buckets)
candidate_hashes = []
for du in [-1, 0, 1]:  # Check adjacent quantization bins
    for dv in [-1, 0, 1]:
        u_nearby = clamp(u_q + du/255, 0, 1)
        v_nearby = clamp(v_q + dv/255, 0, 1)
        h = compute_hash(u_q, v_q, u_nearby, v_nearby)
        candidate_hashes.append(h)

# 3. Bucket lookup (O(1))
candidates = []
for h in candidate_hashes:
    if h in hash_index:
        candidates.extend(hash_index[h])

# 4. Return top K after distance verification
candidates.sort(key=lambda c: distance(query_point, (c.u, c.v)))
return candidates[:k]
```

---

## Three-Tier Architecture

The hash bucketing system is **Tier 1** of a three-tier inference architecture:

### Tier 1: Hash Lookup (O(1), cheapest)

```python
query → hash(u,v) → bucket lookup → ~100 candidates
```

**Cost:** Negligible (hash computation + dict lookup)
**Confidence:** LOW (just spatial proximity)
**Use case:** Fast filtering to reduce search space

### Tier 2: Geometric Verification (cheap)

```python
candidates → compute_geometric_features() → classify relationships
```

**Cost:** 33 geometric features × ~5-20 candidates = ~500 computations (vs 33K for 1K messages)
**Confidence:** HIGH (95.48% variance explained)
**Use case:** Precise relationship classification for narrow-scope decisions

### Tier 3: LLM Inference (expensive)

```python
if curvature > uncertainty_threshold:
    escalate_to_llm(context)
```

**Cost:** Full LLM inference with context
**Confidence:** HIGHEST
**Use case:** Complex reasoning, ambiguous cases, high-curvature semantic regions

**Strategy:** Escalate through tiers only when needed. Most queries stay in Tier 1-2.

---

## Implementation Plan

### Module Structure

**New module:** `luna9-python/luna9/hash_index.py`

```python
class HashIndex:
    """Surface-coordinate hash index for O(1) message retrieval."""

    def __init__(self, bucket_size: int = 100):
        self.buckets: Dict[int, List[MessageEntry]] = {}
        self.bucket_size = bucket_size

    def add_message(self, message_id: str, u: float, v: float,
                   timestamp: datetime) -> int:
        """Add message to index. Returns hash value."""
        pass

    def query(self, u: float, v: float, k: int = 10,
             radius: float = 0.1) -> List[MessageEntry]:
        """Find k nearest messages within radius."""
        pass

    def save(self, path: Path) -> None:
        """Serialize index to disk."""
        pass

    @classmethod
    def load(cls, path: Path) -> 'HashIndex':
        """Deserialize index from disk."""
        pass
```

### Integration with DomainManager

```python
class Domain:
    def __init__(self, ...):
        self.surface = SemanticSurface(...)
        self.hash_index = HashIndex()  # NEW

    def add_message(self, message: str, metadata: Dict = None):
        # Existing: add to surface
        self.surface.append([message])

        # NEW: add to hash index
        u, v = self.surface.project(message)[-1]  # Get coords of new message
        self.hash_index.add_message(
            message_id=len(self.messages) - 1,
            u=u, v=v,
            timestamp=datetime.now()
        )

    def query(self, query_text: str, k: int = 10) -> QueryResult:
        # NEW: Use hash index for candidate retrieval
        embedding = self.surface.model.encode([query_text])[0]
        u, v = self.surface.project_embedding(embedding)

        # Tier 1: Hash lookup
        candidates = self.hash_index.query(u, v, k=k*2)  # Get 2x candidates

        # Tier 2: Geometric verification (if needed)
        # ... compute features and rank

        return QueryResult(...)
```

### Persistence

Hash indices save alongside domain data:

```
~/.luna9/domains/{path}/
    domain.json      - Metadata
    surface.npz      - Embeddings and surface data
    hash_index.pkl   - NEW: Bucket dictionary (pickle or msgpack)
```

---

## Performance Targets

### Current Performance (No Hashing)

From geometric labeling validation:
- 465 message pairs
- 21 minutes feature extraction
- ~2.7 seconds per pair

**Query in 1000-message domain:** ~2.7 seconds × 1000 = **45 minutes** (theoretical worst case)

In practice, queries use vectorized distance calculations (~0.1s for 1000 messages), but geometric feature computation for ranking is still O(N).

### Expected Performance (With Hashing)

**Hash lookup:** < 0.001s (dict lookup)
**Candidate retrieval:** ~20 candidates (instead of N)
**Geometric verification:** 2.7s × 20 = **~0.5 seconds**

**Speedup:** 40x for typical queries
**Scaling:** Query time remains constant as database grows (bucket size is fixed)

### Benchmark Plan

Create `benchmark_hash_index.py`:

```python
# Test scenarios
domains = [
    (100, "Small domain"),
    (1_000, "Medium domain"),
    (10_000, "Large domain"),
    (100_000, "Very large domain")
]

for n_messages, label in domains:
    # Populate domain
    domain = create_test_domain(n_messages)

    # Measure query performance
    queries = sample_queries(n=100)

    # Without hashing
    t1 = time_queries(domain, queries, use_hash=False)

    # With hashing
    t2 = time_queries(domain, queries, use_hash=True)

    print(f"{label}: {t1/t2:.1f}x speedup")
```

**Success criteria:**
- O(1) scaling (query time flat across domain sizes)
- < 1 second query time for domains up to 100K messages
- No degradation in retrieval quality (precision/recall)

---

## Integration with Existing Code

### Minimal Changes Required

**1. SemanticSurface** (add coordinate projection method):
```python
def project_embedding(self, embedding: np.ndarray) -> Tuple[float, float]:
    """Project embedding to (u, v) surface coordinates."""
    # Use existing Bézier surface math
    return u, v
```

**2. Domain** (add hash index):
```python
def __init__(self, ...):
    # ... existing initialization
    self.hash_index = HashIndex() if use_hash else None
```

**3. DomainManager** (save/load hash indices):
```python
def save_domain(self, path: str) -> Dict:
    # ... existing save logic
    if domain.hash_index:
        domain.hash_index.save(storage_path / "hash_index.pkl")
```

### Backward Compatibility

- Hash index is optional (controlled by flag)
- Existing domains work without modification
- Hash index can be built lazily from existing domain data

---

## Open Questions

### 1. Quantization Resolution

**Current:** 8 bits per coordinate (256 bins)
**Trade-off:** More bits = more precision but larger hash space

Should we make this configurable? Or adaptive based on domain size?

### 2. Bucket Size Tuning

**Current:** 100 messages per bucket
**Question:** Optimal bucket size may depend on:
- Domain size
- Query patterns
- Memory constraints

Should we auto-tune or keep it fixed?

### 3. Multi-Resolution Hashing

Should we use multiple quantization levels? Example:
- Coarse hash (4 bits) → rough filtering
- Fine hash (8 bits) → precise bucketing

Could improve recall for edge cases where messages fall between quantization bins.

### 4. Hash Index Rebuild Strategy

When should we rebuild the index?
- After N new messages?
- When bucket distribution becomes imbalanced?
- On explicit request only?

---

## Next Steps

### Phase 2A Implementation

1. ✅ Document algorithm (this file)
2. 📋 Implement `luna9.hash_index` module
3. 📋 Integrate with `SemanticSurface` (add projection method)
4. 📋 Update `Domain` to use hash index
5. 📋 Update `DomainManager` save/load
6. 📋 Write unit tests
7. 📋 Create benchmark suite
8. 📋 Validate performance targets

### Future Enhancements (Phase 3+)

- **Adaptive quantization** - Adjust resolution based on domain size
- **Multi-resolution indexing** - Hierarchical hash levels
- **Distributed hashing** - Shard large domains across multiple indices
- **Approximate nearest neighbors** - Integrate with libraries like FAISS for hybrid approach

---

## References

- **audfprint:** [github.com/dpwe/audfprint](https://github.com/dpwe/audfprint) - Audio fingerprinting inspiration
- **Geometric Relationship Inference:** `docs/geometric_relationship_inference.md` - Foundation for Tier 2 verification
- **Semantic Surfaces:** Core Luna Nine architecture using Bézier surfaces for semantic memory

---

*Part of the Luna Nine project - geometric memory and initiative for AI systems*
