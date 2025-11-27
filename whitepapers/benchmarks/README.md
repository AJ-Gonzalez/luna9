# Parametric Surfaces Whitepaper Benchmarks

This folder contains the benchmarking infrastructure for the "Parametric Surfaces for Semantic Memory" whitepaper.

## Files

### Benchmark Scripts
- **`benchmark_parametric_surfaces.py`** - Main benchmark harness
  - Tests Luna9 retrieval quality (Recall@k, Precision@k, MRR)
  - Measures query performance (latency, memory)
  - Runs at multiple scales (100, 500, 1000 chunks)

- **`gutenberg_loader.py`** - Project Gutenberg corpus loader
  - Downloads public domain books
  - Chunks using Luna9's smart chunker
  - Creates ground truth query-passage pairs

- **`fill_whitepaper_results.py`** - Results integration script
  - Reads benchmark JSON results
  - Fills placeholders in whitepaper
  - Formats tables and statistics

### Results Data
- **`benchmark_results_100.json`** - 100-chunk corpus results
- **`benchmark_results_500.json`** - 500-chunk corpus results
- **`benchmark_results_1000.json`** - 1000-chunk corpus results

### Cache
- **`gutenberg_cache/`** - Cached Project Gutenberg downloads

## Running Benchmarks

```bash
# Run full benchmark suite
python benchmark_parametric_surfaces.py

# Fill whitepaper with results
python fill_whitepaper_results.py
```

## Key Findings

**Retrieval Quality:**
- Recall@20: 10.0% on thematic queries (100 chunks)
- MRR: 0.053

**Performance:**
- Query latency: 151ms → 496ms (100 → 1000 chunks)
- Sub-linear scaling: 10x data → 3.3x latency

**Memory:**
- 2.3 KB per message at 1000-chunk scale
- Fixed overhead amortizes with corpus size

## Methodology

**Pure retrieval benchmarks** - decouples retrieval quality from LLM synthesis quality.

We measure whether Luna9 retrieves the right chunks, not whether models use them well.

Ground truth queries are thematic ("passages about marriage") with labeled relevant passages.
