# Luna9 Whitepapers

Research papers exploring the mathematical foundations of Luna9's semantic memory system.

## Published Papers

### 1. Parametric Surfaces for Semantic Memory ✓
**File:** [`01_parametric_surfaces.md`](01_parametric_surfaces.md)

Modeling semantic space as a parametric Bézier surface enables sub-linear retrieval (O(√n)) while preserving geometric properties. Includes benchmarks on Project Gutenberg corpus.

**Key Results:**
- Query latency: 151ms → 496ms (100 → 1000 chunks)
- Sub-linear scaling: 10x data → 3.3x latency
- Memory: 2.3 KB per message at scale

## Roadmap

See [`INDEX.md`](INDEX.md) for the complete list of planned whitepapers covering:
- Curvature as semantic importance
- Flow suppression for dispersed retrieval
- Dual-mode retrieval strategies
- Spatial hashing for O(1) lookups
- And more...

## Structure

```
whitepapers/
├── 01_parametric_surfaces.md    # Published paper
├── INDEX.md                      # Complete roadmap
├── benchmarks/                   # Benchmark code & data
│   ├── README.md
│   ├── benchmark_parametric_surfaces.py
│   ├── gutenberg_loader.py
│   └── benchmark_results_*.json
└── archive/                      # Old drafts & planning
```

## Running Benchmarks

```bash
cd benchmarks
python benchmark_parametric_surfaces.py
```

See [`benchmarks/README.md`](benchmarks/README.md) for details.

## Contributing

These whitepapers are research artifacts documenting Luna9's mathematical foundations. If you find errors or have suggestions, please open an issue.
