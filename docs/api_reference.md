# Luna Nine API Reference

**Version:** 0.1.0
**Last Updated:** November 2025

## Overview

This document provides the API reference for Luna Nine's core components: domain management, geometric memory, and relationship inference.

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```python
from luna9 import DomainManager, DomainType

# Initialize domain manager
manager = DomainManager("./memory")

# Create a domain
manager.create_domain("conversations/work", DomainType.PROJECT)

# Add messages
messages = [
    {"text": "How do I handle authentication?", "speaker": "user", "metadata": {}},
    {"text": "Use JWT tokens for stateless auth", "speaker": "assistant", "metadata": {}}
]
manager.add_to_domain("conversations/work", messages)

# Search semantically
results = manager.search_domain(
    "conversations/work",
    query="authentication security",
    mode="semantic",
    k=5
)
```

---

## Domain Management

### `DomainManager`

Primary interface for managing knowledge domains.

```python
class DomainManager:
    def __init__(self, base_path: str, encoder_model: str = "all-MiniLM-L6-v2")
```

**Parameters:**
- `base_path` (str): Root directory for domain storage
- `encoder_model` (str): Sentence transformer model for embeddings

**Example:**
```python
manager = DomainManager("./memory")
```

---

### Creating Domains

```python
def create_domain(
    self,
    domain_path: str,
    domain_type: DomainType,
    parent_domain: Optional[str] = None,
    metadata: Optional[Dict] = None
) -> None
```

Create a new knowledge domain.

**Parameters:**
- `domain_path` (str): Domain identifier (e.g., "work/project_x")
- `domain_type` (DomainType): Type of domain (PROJECT, FOUNDATION, REFERENCE, TEMPORARY)
- `parent_domain` (Optional[str]): Parent domain path for hierarchy
- `metadata` (Optional[Dict]): Additional domain metadata

**Raises:**
- `DomainExistsError`: Domain already exists
- `DomainHierarchyError`: Invalid parent or nesting (max 3 levels)

**Example:**
```python
manager.create_domain("books/rust", DomainType.REFERENCE)
manager.create_domain("books/rust/ownership", DomainType.REFERENCE, parent_domain="books/rust")
```

---

### Adding Messages

```python
def add_to_domain(
    self,
    domain_path: str,
    messages: List[Dict[str, Any]],
    auto_rebuild: bool = True
) -> None
```

Add messages to a domain.

**Parameters:**
- `domain_path` (str): Target domain
- `messages` (List[Dict]): Messages to add. Each message should have:
  - `text` (str, required): Message content
  - `speaker` (str, optional): Who said it (default: "unknown")
  - `metadata` (dict, optional): Additional metadata
- `auto_rebuild` (bool): Rebuild surface immediately (default: True)

**Message format:**
```python
{
    "text": "Message content",
    "speaker": "user",  # or "assistant", "system", etc.
    "metadata": {
        "timestamp": "2025-11-20T10:30:00",
        "source": "github.com/user/repo",
        "confidence": 0.95
    }
}
```

**Example:**
```python
messages = [
    {
        "text": "What's the best way to handle errors in Rust?",
        "speaker": "user"
    },
    {
        "text": "Use Result<T, E> for recoverable errors and panic! for unrecoverable ones",
        "speaker": "assistant",
        "metadata": {"source": "rust-book", "chapter": 9}
    }
]
manager.add_to_domain("books/rust", messages)
```

---

### Searching Domains

```python
def search_domain(
    self,
    domain_path: str,
    query: str,
    mode: str = "semantic",
    k: int = 5,
    search_descendants: bool = True
) -> Dict[str, Any]
```

Search within a domain.

**Parameters:**
- `domain_path` (str): Domain to search
- `query` (str): Search query
- `mode` (str): Search mode - "semantic", "literal", or "both"
- `k` (int): Number of results to return
- `search_descendants` (bool): Include child domains in search

**Returns:**
```python
{
    "domain": str,  # Domain path
    "query": str,   # Original query
    "mode": str,    # Search mode used
    "messages": List[Dict],  # Matched messages with scores
    "execution_time": float  # Search time in seconds
}
```

**Message result format:**
```python
{
    "text": str,
    "speaker": str,
    "score": float,  # Relevance score (0-1)
    "metadata": dict,
    "index": int  # Position in domain
}
```

**Example:**
```python
# Semantic search
results = manager.search_domain(
    "books/rust",
    query="memory safety",
    mode="semantic",
    k=3
)

for msg in results["messages"]:
    print(f"Score: {msg['score']:.3f}")
    print(f"Text: {msg['text'][:100]}")
    print(f"Metadata: {msg['metadata']}")
```

---

### Listing Domains

```python
def list_domains(self, prefix: Optional[str] = None) -> List[Dict[str, Any]]
```

List all domains, optionally filtered by prefix.

**Parameters:**
- `prefix` (Optional[str]): Filter by domain path prefix

**Returns:** List of domain info dictionaries

**Example:**
```python
# All domains
all_domains = manager.list_domains()

# Just book-related domains
book_domains = manager.list_domains(prefix="books")
```

---

### Domain Info

```python
def get_domain_info(self, domain_path: str) -> Dict[str, Any]
```

Get detailed information about a domain.

**Returns:**
```python
{
    "domain_path": str,
    "domain_type": str,
    "message_count": int,
    "created_at": str,
    "last_modified": str,
    "parent": Optional[str],
    "children": List[str],
    "metadata": dict,
    "is_loaded": bool
}
```

**Example:**
```python
info = manager.get_domain_info("books/rust")
print(f"Messages: {info['message_count']}")
print(f"Children: {info['children']}")
```

---

### Loading/Unloading Domains

```python
def load_domain(self, domain_path: str) -> None
def unload_domain(self, domain_path: str) -> None
```

Manually load or unload domains from memory.

**Note:** Domains are automatically loaded on first access and can be unloaded to free memory.

---

### Deleting Domains

```python
def delete_domain(
    self,
    domain_path: str,
    delete_descendants: bool = False
) -> None
```

Delete a domain and optionally its children.

**Parameters:**
- `domain_path` (str): Domain to delete
- `delete_descendants` (bool): Also delete child domains

**Raises:**
- `DomainNotFoundError`: Domain doesn't exist
- `DomainHierarchyError`: Has children but `delete_descendants=False`

---

## Geometric Surfaces

### `SemanticSurface`

Represents a Bézier surface encoding semantic relationships.

```python
from luna9 import SemanticSurface

surface = SemanticSurface.create_from_messages(
    name="my_surface",
    messages=[{"text": "..."}]
)
```

---

### Query

```python
def query(
    self,
    query_text: str,
    k: int = 5,
    mode: str = "semantic"
) -> Dict[str, Any]
```

Search the surface for relevant messages.

**Parameters:**
- `query_text` (str): Search query
- `k` (int): Number of results
- `mode` (str): "semantic", "literal", or "both"

**Returns:**
```python
{
    "query": str,
    "mode": str,
    "messages": List[Dict],  # Top k results
    "sources": {
        "embeddings": ndarray,
        "messages": List[Dict]
    }
}
```

---

### Append

```python
def append(self, messages: List[Dict[str, Any]]) -> None
```

Add messages to surface (lazy rebuild).

**Example:**
```python
surface.append([
    {"text": "New message", "speaker": "user"}
])
```

---

### Save/Load

```python
def save(self, path: str) -> None
@classmethod
def load(cls, path: str) -> "SemanticSurface"
```

Persist and restore surfaces.

**Example:**
```python
surface.save("./memory/my_surface")
loaded = SemanticSurface.load("./memory/my_surface")
```

---

## Geometric Relationship Inference

### `GeometricLabelingPipeline`

End-to-end pipeline for inferring semantic relationships.

```python
from luna9.labeling import GeometricLabelingPipeline

pipeline = GeometricLabelingPipeline(
    parquet_dir="./data",
    output_dir="./results",
    encoder_model="all-MiniLM-L6-v2"
)

results = pipeline.run_full_pipeline(
    sample_size=500,
    pair_strategy="mixed",
    max_pairs=1000,
    n_clusters=5
)
```

---

### `GeometricFeatureExtractor`

Extract geometric features from message pairs.

```python
from luna9.labeling import GeometricFeatureExtractor

extractor = GeometricFeatureExtractor(
    surface=semantic_surface,
    embeddings=embeddings_dict
)

features = extractor.extract_pair_features(
    msg1="First message",
    msg2="Second message"
)
```

**Features extracted (33 total):**
- Distance metrics (3)
- Path curvature (8)
- Surface topology (10)
- Tangent alignment (6)
- Influence overlap (6)

---

### `cluster_by_geometry`

Cluster message pairs by geometric similarity.

```python
from luna9.labeling import cluster_by_geometry

clusters, labels = cluster_by_geometry(
    features_df,
    n_clusters=5,
    method="kmeans"
)
```

**Parameters:**
- `features_df` (DataFrame): Extracted geometric features
- `n_clusters` (int): Number of clusters
- `method` (str): "kmeans" or "dbscan"

**Returns:**
- `clusters` (ndarray): Cluster assignments
- `labels` (Dict): Inferred relationship types per cluster

---

## Exceptions

### `DomainError`

Base exception for all domain-related errors.

### `DomainNotFoundError`

Domain doesn't exist.

```python
try:
    manager.search_domain("nonexistent", query="test")
except DomainNotFoundError as e:
    print(f"Domain not found: {e}")
```

### `DomainExistsError`

Domain already exists.

```python
try:
    manager.create_domain("existing", DomainType.PROJECT)
except DomainExistsError:
    print("Domain already exists")
```

### `DomainHierarchyError`

Invalid hierarchy operation (e.g., too deep, orphaned children).

### `SearchError`

Search operation failed.

---

## Enums

### `DomainType`

```python
from luna9 import DomainType

DomainType.PROJECT      # Project-specific knowledge
DomainType.FOUNDATION   # Foundational concepts
DomainType.REFERENCE    # Reference material (books, docs)
DomainType.TEMPORARY    # Short-lived, can be deleted
```

### `SemanticRelationship`

```python
from luna9.labeling import SemanticRelationship

SemanticRelationship.NECESSARY     # Direct Q&A, prerequisite
SemanticRelationship.OPPOSED       # Contradictory, conflicting
SemanticRelationship.ANCILLARY     # Supporting, optional context
SemanticRelationship.HIERARCHICAL  # General to specific
SemanticRelationship.IRRELEVANT    # Unrelated
```

---

## Configuration

### Environment Variables

```bash
# Model cache directory
export TRANSFORMERS_CACHE=/path/to/cache

# Device (cuda, cpu)
export TORCH_DEVICE=cuda

# Logging level
export LUNA9_LOG_LEVEL=INFO
```

### Custom Encoder Models

Any sentence-transformers compatible model:

```python
manager = DomainManager(
    "./memory",
    encoder_model="sentence-transformers/all-mpnet-base-v2"
)
```

Popular models:
- `all-MiniLM-L6-v2` (default, fast, 384 dim)
- `all-mpnet-base-v2` (better quality, 768 dim)
- `multi-qa-mpnet-base-dot-v1` (optimized for Q&A)

---

## Performance Considerations

### Memory Usage

Approximate memory per domain:
```
messages * (embedding_size * 4 bytes + text_size)
```

Example: 10K messages, 384-dim embeddings:
```
10,000 * (384 * 4 + ~500) = ~20 MB
```

### Search Performance

Current (linear search):
- 1K messages: ~10ms
- 10K messages: ~50ms
- 100K messages: ~500ms

Planned (hash index):
- Any size: ~1-10ms

### Optimization Tips

1. **Unload unused domains:**
```python
manager.unload_domain("temporary/old_project")
```

2. **Use smaller embedding models for large-scale:**
```python
manager = DomainManager("./memory", encoder_model="all-MiniLM-L6-v2")
```

3. **Lazy rebuilding:**
```python
manager.add_to_domain(domain, messages, auto_rebuild=False)
# ... add more messages ...
manager.load_domain(domain)  # Rebuilds once
```

---

## Examples

### Example 1: Knowledge Base

```python
from luna9 import DomainManager, DomainType

manager = DomainManager("./kb")

# Create domain hierarchy
manager.create_domain("docs/api", DomainType.REFERENCE)
manager.create_domain("docs/tutorials", DomainType.REFERENCE)

# Add API documentation
api_docs = [
    {"text": "GET /users - List all users", "metadata": {"endpoint": "users"}},
    {"text": "POST /users - Create a user", "metadata": {"endpoint": "users"}},
]
manager.add_to_domain("docs/api", api_docs)

# Search across all docs
results = manager.search_domain(
    "docs",
    query="create new user",
    search_descendants=True,
    k=3
)
```

### Example 2: Conversation Memory

```python
# Store conversation turns
conversation = [
    {"text": "I need to debug a memory leak", "speaker": "user"},
    {"text": "Let's check for circular references first", "speaker": "assistant"},
    {"text": "Found it! The event listener wasn't cleaned up", "speaker": "user"},
]

manager.create_domain("sessions/2025-11-20", DomainType.TEMPORARY)
manager.add_to_domain("sessions/2025-11-20", conversation)

# Later, recall similar issues
results = manager.search_domain(
    "sessions",
    query="memory leak debugging",
    search_descendants=True
)
```

### Example 3: Geometric Relationship Check

```python
from luna9.labeling import GeometricFeatureExtractor

# Extract features between two messages
features = extractor.extract_pair_features(
    msg1="Use async/await for concurrent operations",
    msg2="Never use threads in Python"
)

# Check if opposed
if features["cosine_similarity"] < 0 and features["total_path_curvature"] > 0.5:
    print("Messages appear contradictory")
```

---

## Contributing

API improvements welcome! Please:
1. Maintain backward compatibility
2. Add docstrings
3. Include type hints
4. Add examples

See main README for contribution guidelines.

---

## Changelog

### 0.1.0 (November 2025)
- Initial API release
- Domain management system
- Geometric surfaces
- Relationship inference (POC)
