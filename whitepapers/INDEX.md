# Luna9 Whitepapers

This series documents the core mathematical ideas behind Luna9's geometric approach to semantic memory.

---

## Papers

### 1. Parametric Surfaces for Semantic Space ✓ COMPLETE
**File:** `01_parametric_surfaces.md`

**Status:** Full whitepaper with benchmarks on Project Gutenberg corpus

**Summary:**
- Treat embeddings as control points for a smooth parametric surface (Bézier)
- Navigate semantic space using UV coordinates (0-1 range)
- Enables sub-linear query performance (O(√n) to O(n^0.7))
- Linear storage overhead (1.54 KB per message)

**Analogies:**
- Game engines use parametric surfaces for terrain
- CAD software uses them for curves and surfaces
- Luna9 applies the same mathematics to embedding space

---

### 2. Curvature as Semantic Importance
**File:** `02_curvature_semantics.md` *(planned)*

**Summary:**
- High curvature indicates semantic transition points
- Sharp surface bends correspond to shifts in meaning
- Identifies junctions, decision points, and narrative turns
- Low curvature indicates steady conceptual flow

**Applications:**
- Find important moments without keyword search
- Detect narrative structure geometrically
- Navigate to semantically significant regions

---

### 3. Normal Vectors as Flow Direction
**File:** `03_flow_suppression.md` *(planned)*

**Summary:**
- Surface normals indicate direction of strongest semantic flow
- Strong flow can dominate retrieval, hiding dispersed signals
- Suppression technique reveals scattered content

**Mathematics:**
- Normal: n = embedding - surface_point
- Suppression: n' = n - f(||n||) * (n/||n||)
- Power law: f(m) = m^β where β > 1

**Applications:**
- Solves cohesive flow domination problem
- Similar to dynamic range compression in audio
- Finds scattered content missed by vector search

---

### 4. Dual-Mode Retrieval
**File:** `04_dual_mode_retrieval.md` *(planned)*

**Summary:**
- Combines surface navigation (cohesive content) with flow suppression (dispersed content)
- Surface mode: storylines, explanations, contextual flow
- Suppression mode: scattered mentions, entities across chapters
- Provides comprehensive retrieval coverage

---

### 5. Spatial Hashing for Sub-linear Performance
**File:** `05_spatial_hashing.md` *(planned)*

**Summary:**
- UV space is 2D (coordinates in 0-1 range)
- Divide into grid cells for spatial indexing
- Query only nearby cells: O(1) candidate retrieval
- Enables scaling to millions of messages

**Implementation:**
- Based on spatial hashing techniques from collision detection
- Minimal memory overhead
- Standard spatial indexing approach

---

### 6. Semantic Chunking
**File:** `06_smart_chunking.md` *(planned)*

**Summary:**
- Respects paragraph and sentence boundaries
- Includes overlap between chunks for continuity
- Preserves semantic coherence

**Results:**
- 0% mid-sentence breaks vs 27% with character-based chunking
- Improved downstream retrieval quality

---

### 7. LMIX Translation Layer
**File:** `07_lmix_translation.md` *(planned)*

**Summary:**
- Translates geometric properties to natural language
- Enables LLM interpretation of geometric context
- No model fine-tuning required

**Example:**
- Instead of: "curvature=0.82"
- Output: "You are at a junction (high curvature)"

---

### 8. Initiative from Conditions
**File:** `08_initiative_from_conditions.md` *(planned)*

**Research Question:**
- Can AI initiative emerge from geometric context rather than programmatic triggers?
- Provide State + Possibilities + Boundaries as natural language
- Test whether this produces more coherent autonomous behavior

**Status:**
- Early research phase
- Requires further validation
- Most speculative component

---

## Benchmark Data

Full benchmark harness and results available in `benchmarks/` directory.
