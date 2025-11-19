# Proof of Concept: Path Curvature as Semantic Classifier

This validates that **path curvature on Bézier surfaces can classify semantic relationships**.

## Quick Start

**Install dependencies:**
```bash
pip install -r requirements.txt
```

**Run the validation:**
```bash
python validate_synthetic.py
```

**What you'll see:**
- Analysis of 40 labeled message pairs across 5 domains
- Path curvature measurements for each relationship type
- Statistical summary confirming: **Direct < Related < Distant**

Current validation results:
- **Direct** (25 pairs): mean = 0.015 radians (tight semantic connection)
- **Related** (9 pairs): mean = 0.023 radians (bridging concepts)
- **Distant** (6 pairs): mean = 0.025 radians (requires context switching)

**Hypothesis confirmed** across code review, cooking, science, travel, and book analysis domains.

## Understanding Path Curvature

**What it measures:** The geometric "cost" of moving between two points on a semantic surface.

- **Low curvature** = straight path = direct semantic connection (Q&A pairs, immediate elaborations)
- **Medium curvature** = gentle curves = related concepts (same topic, building on ideas)
- **High curvature** = winding path = distant relationships (requires bridging, context shifts)

**Why it works:** Semantic surfaces are Bézier surfaces with messages as control points. The surface geometry naturally encodes semantic structure, making path curvature a proxy for relationship complexity.

## Test With Your Own Data

Want to validate (or disprove) this with your own conversations?

### 1. Create a conversation file

Format: `data/my_conversation.txt`
```
First message here.

Second message here.

Third message here.
```

Messages separated by double newlines (`\n\n`). Start with 8-12 messages for quick testing.

### 2. Label message pairs

Edit `data/synthetic_labeled_relationships.json`:

```json
{
  "metadata": {
    "domains": {
      "my_domain": "my_conversation.txt (10 messages)"
    }
  },
  "labeled_pairs": [
    {
      "domain": "my_domain",
      "idx1": 0,
      "idx2": 1,
      "label": "direct",
      "note": "Question and direct answer"
    },
    {
      "domain": "my_domain",
      "idx1": 0,
      "idx2": 5,
      "label": "related",
      "note": "Same topic, different aspect"
    },
    {
      "domain": "my_domain",
      "idx1": 0,
      "idx2": 8,
      "label": "distant",
      "note": "Topic shift to tangentially related concept"
    }
  ]
}
```

**Labels:**
- `"direct"` - Q&A pairs, immediate elaborations, direct follow-ups
- `"related"` - Same topic area, building on concepts, thematic connections
- `"distant"` - Topic shifts, tangential relationships, requires bridging

**Tips:**
- Label at least 3-4 pairs per type
- Use message indices starting from 0
- Include your reasoning in the `"note"` field
- Mix different relationship types to test the hypothesis

### 3. Run validation

```bash
python validate_synthetic.py
```

Results get saved to `data/synthetic_validation_results.json` and printed to console.

## What's Here

**Validation & Testing:**
- `validate_synthetic.py` - Main validation script
- `generate_synthetic_data.py` - Creates synthetic conversation data
- `test_surface.py` - Unit tests for surface math
- `test_semantic_surface.py` - Integration tests

**Demos (explore the concepts):**
- `demo_path_curvature.py` - Visual walkthrough of curvature analysis
- `demo_dual_retrieval.py` - Shows smooth + exact retrieval modes
- `demo_comparison.py` - Semantic surface vs baseline cosine similarity

**Core Implementation:**
- `semantic_surface.py` - Semantic surface with message storage
- `surface_math.py` - Bézier surface math, curvature computation
- `baseline.py` - Traditional cosine similarity baseline

**Data:**
- `data/synthetic_*.txt` - 5 domain conversations (88 messages total)
- `data/synthetic_labeled_relationships.json` - Ground truth labels (40 pairs)
- `data/synthetic_validation_results.json` - Validation output

## Dependencies

```
numpy>=1.24.0
scipy>=1.10.0
sentence-transformers>=2.2.0
scikit-learn>=1.3.0
pytest>=7.4.0
```

First run downloads the embedding model (~100MB).

## Results Interpretation

If the hypothesis holds for your data:
- `mean(direct) < mean(related) < mean(distant)`
- Path curvature successfully classifies semantic relationships
- Geometry encodes meaning

If it doesn't:
- That's valuable data! Different conversation styles might need different surface structures
- Consider domain characteristics (technical vs casual, Q&A vs narrative)
- Open an issue and share your findings

## What This Enables

If path curvature reliably classifies semantic relationships, we can:
- Use geometry for memory management (high curvature = important transitions)
- Navigate semantic space instead of searching it
- Preserve relationship topology when compressing context
- Build multi-surface awareness systems

See the main [README](../README.md) for the full vision.
