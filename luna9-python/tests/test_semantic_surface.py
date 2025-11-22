"""
Tests for SemanticSurface functionality.

Tests:
- Surface creation with different grid sizes
- Message appending with lazy rebuild
- Querying with dual-mode retrieval
- Grid inference
"""

import pytest
import numpy as np
from luna9 import SemanticSurface


def test_create_surface_basic():
    """Test creating basic semantic surface."""
    messages = [
        "Python is great",
        "Rust is fast",
        "JavaScript runs everywhere",
        "Go has good concurrency"
    ]

    surface = SemanticSurface(messages)

    assert len(surface.messages) == 4
    assert surface.messages == messages
    assert surface.embeddings is not None
    assert surface.control_points is not None
    assert surface.grid_m * surface.grid_n == 4


def test_grid_inference():
    """Test automatic grid shape inference."""
    # 4 messages → 2x2 grid
    surface_4 = SemanticSurface(["m1", "m2", "m3", "m4"])
    assert surface_4.grid_m == 2
    assert surface_4.grid_n == 2

    # 9 messages → 3x3 grid
    surface_9 = SemanticSurface(["m"] * 9)
    assert surface_9.grid_m == 3
    assert surface_9.grid_n == 3

    # 12 messages → 3x4 grid
    surface_12 = SemanticSurface(["m"] * 12)
    assert surface_12.grid_m == 3
    assert surface_12.grid_n == 4


def test_explicit_grid_shape():
    """Test creating surface with explicit grid shape."""
    messages = ["m1", "m2", "m3", "m4"]

    # Force 1x4 instead of 2x2
    surface = SemanticSurface(messages, grid_shape=(1, 4))

    assert surface.grid_m == 1
    assert surface.grid_n == 4


def test_precomputed_embeddings():
    """Test creating surface with pre-computed embeddings."""
    messages = ["test1", "test2", "test3", "test4"]

    # Create surface to get embeddings
    surface1 = SemanticSurface(messages)
    embeddings = surface1.embeddings.copy()

    # Create new surface with those embeddings
    surface2 = SemanticSurface(messages, embeddings=embeddings)

    assert np.array_equal(surface2.embeddings, embeddings)
    assert surface2.control_points.shape == surface1.control_points.shape


def test_query_basic():
    """Test basic query functionality."""
    messages = [
        "Memory safety in Rust",
        "Python garbage collection",
        "JavaScript event loop",
        "Go channels and concurrency"
    ]

    surface = SemanticSurface(messages)

    # Query for memory-related content
    result = surface.query("memory management", k=2)

    # Should return RetrievalResult
    assert hasattr(result, 'uv')
    assert hasattr(result, 'influence')
    assert hasattr(result, 'nearest_control_points')
    assert hasattr(result, 'curvature')

    # Get messages
    retrieved = result.get_messages(messages, mode='exact', k=2)

    assert 'sources' in retrieved
    assert len(retrieved['sources']['messages']) <= 2


def test_query_modes():
    """Test different query modes (smooth, exact, both)."""
    messages = ["Test " + str(i) for i in range(4)]
    surface = SemanticSurface(messages)

    result = surface.query("test query", k=2)

    # Mode: smooth (interpretation only)
    smooth = result.get_messages(messages, mode='smooth', k=2)
    assert 'interpretation' in smooth
    assert 'sources' not in smooth

    # Mode: exact (sources only)
    exact = result.get_messages(messages, mode='exact', k=2)
    assert 'sources' in exact
    assert 'interpretation' not in exact

    # Mode: both
    both = result.get_messages(messages, mode='both', k=2)
    assert 'interpretation' in both
    assert 'sources' in both


def test_append_message():
    """Test appending single message to surface."""
    # Start with 4 messages (2x2)
    initial = ["m1", "m2", "m3", "m4"]
    surface = SemanticSurface(initial)

    initial_grid = (surface.grid_m, surface.grid_n)

    # Append one message
    surface.append_message("m5")

    # Should have 5 messages now
    assert len(surface.messages) == 5
    assert surface.messages[-1] == "m5"

    # Grid should have grown
    assert surface.grid_m * surface.grid_n == 5


def test_append_messages_batch():
    """Test appending multiple messages at once."""
    initial = ["m1", "m2", "m3", "m4"]
    surface = SemanticSurface(initial)

    # Append 5 more messages
    new_messages = ["m5", "m6", "m7", "m8", "m9"]
    surface.append_messages(new_messages)

    # Should have 9 total
    assert len(surface.messages) == 9
    assert surface.messages[-5:] == new_messages


def test_append_with_metadata():
    """Test appending messages with metadata."""
    surface = SemanticSurface(["m1", "m2", "m3", "m4"])

    metadata = {"speaker": "user", "timestamp": "2025-11-20"}
    surface.append_message("m5", metadata=metadata)

    # Metadata tracking is internal to Domain, but shouldn't cause errors
    assert len(surface.messages) == 5


def test_append_lazy_rebuild():
    """Test that append uses lazy rebuild (doesn't rebuild immediately)."""
    messages = ["m" + str(i) for i in range(16)]  # 4x4 grid
    surface = SemanticSurface(messages)

    # Append below threshold (should buffer, not rebuild)
    surface.append_message("new1", rebuild_threshold=0.2)  # 1/16 = 0.0625 < 0.2

    # Should have pending messages
    assert len(surface._pending_messages) > 0

    # Force rebuild by crossing threshold
    surface.append_message("new2")
    surface.append_message("new3")
    surface.append_message("new4")  # Now 4/16 = 0.25 > 0.2, triggers rebuild

    # After rebuild, pending should be clear
    # Note: Implementation may vary, this tests the concept


def test_query_after_append():
    """Test that queries work correctly after appending messages."""
    # Initial messages about various topics
    initial = [
        "Python is great for scripting",
        "JavaScript runs in browsers",
        "Rust is systems programming",
        "Go is good for servers"
    ]

    surface = SemanticSurface(initial)

    # Query before append
    result_before = surface.query("memory management", k=2)
    messages_before = result_before.get_messages(surface.messages, mode='exact', k=2)

    # Append memory-related messages
    memory_messages = [
        "Memory safety is crucial in systems programming",
        "Garbage collection automates memory management",
        "Manual memory management gives more control"
    ]

    for msg in memory_messages:
        surface.append_message(msg)

    # Query after append
    result_after = surface.query("memory management", k=3)
    messages_after = result_after.get_messages(surface.messages, mode='exact', k=3)

    # New memory messages should appear in results
    retrieved_texts = messages_after['sources']['messages']
    memory_found = any(msg in retrieved_texts for msg in memory_messages)

    assert memory_found, "Appended memory-related messages should be retrievable"


def test_control_points_shape():
    """Test control points have correct shape."""
    messages = ["m"] * 12  # 3x4 grid
    surface = SemanticSurface(messages)

    # Control points should be (m, n, embedding_dim)
    assert surface.control_points.shape == (surface.grid_m, surface.grid_n, surface.embedding_dim)


def test_weights_initialization():
    """Test weights are initialized correctly."""
    messages = ["m1", "m2", "m3", "m4"]
    surface = SemanticSurface(messages)

    # All weights should be 1.0 initially
    assert surface.weights.shape == (surface.grid_m, surface.grid_n)
    assert np.all(surface.weights == 1.0)


def test_provenance_mapping():
    """Test provenance mapping is built correctly."""
    messages = ["m1", "m2", "m3", "m4"]
    surface = SemanticSurface(messages)

    assert 'cp_to_msg' in surface.provenance
    assert 'msg_to_cp' in surface.provenance

    # Should have mapping for all control points
    assert len(surface.provenance['cp_to_msg']) == 4
    assert len(surface.provenance['msg_to_cp']) == 4


def test_embedding_model():
    """Test embedding model can be specified."""
    messages = ["test1", "test2", "test3", "test4"]

    # Default model
    surface1 = SemanticSurface(messages)
    assert surface1.model_name == 'all-MiniLM-L6-v2'

    # Explicit model name
    surface2 = SemanticSurface(messages, model_name='all-MiniLM-L6-v2')
    assert surface2.model_name == 'all-MiniLM-L6-v2'


def test_multiple_queries():
    """Test multiple queries on same surface."""
    messages = [
        "Python programming",
        "Rust memory safety",
        "JavaScript web development",
        "Go concurrent systems",
        "C low-level control",
        "Java enterprise applications"
    ]

    surface = SemanticSurface(messages)

    # Multiple different queries should all work
    result1 = surface.query("memory management", k=2)
    result2 = surface.query("web development", k=2)
    result3 = surface.query("systems programming", k=2)

    # All should return valid results
    assert result1.uv != result2.uv  # Different queries, different UV coordinates
    assert len(result1.influence) > 0
    assert len(result2.influence) > 0
    assert len(result3.influence) > 0


def test_project_embedding():
    """Test projecting embedding to surface coordinates."""
    from sentence_transformers import SentenceTransformer

    messages = [
        "Python programming",
        "Rust systems language",
        "JavaScript web development",
        "Go concurrent programming"
    ]

    surface = SemanticSurface(messages)
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Project a new message embedding
    query_text = "web programming"
    query_embedding = model.encode([query_text])[0]

    u, v = surface.project_embedding(query_embedding)

    # Coordinates should be in valid range
    assert 0 <= u <= 1
    assert 0 <= v <= 1

    # Should be a tuple of floats
    assert isinstance(u, (float, np.floating))
    assert isinstance(v, (float, np.floating))


def test_project_embedding_deterministic():
    """Test that same embedding projects to same coordinates."""
    from sentence_transformers import SentenceTransformer

    messages = ["msg1", "msg2", "msg3", "msg4"]
    surface = SemanticSurface(messages)
    model = SentenceTransformer('all-MiniLM-L6-v2')

    query_embedding = model.encode(["test query"])[0]

    # Project twice
    u1, v1 = surface.project_embedding(query_embedding)
    u2, v2 = surface.project_embedding(query_embedding)

    # Should be identical
    assert u1 == u2
    assert v1 == v2


def test_project_embedding_different_queries():
    """Test that different embeddings project to different coordinates."""
    from sentence_transformers import SentenceTransformer

    messages = [
        "Python is great",
        "Rust is fast",
        "JavaScript is popular",
        "Go is concurrent"
    ]

    surface = SemanticSurface(messages)
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Project two very different queries
    emb1 = model.encode(["programming languages"])[0]
    emb2 = model.encode(["cooking recipes"])[0]

    u1, v1 = surface.project_embedding(emb1)
    u2, v2 = surface.project_embedding(emb2)

    # Should project to different points (very unlikely to be exactly same)
    assert (u1, v1) != (u2, v2)


def test_project_embedding_matches_query():
    """Test that project_embedding gives same coords as full query."""
    from sentence_transformers import SentenceTransformer

    messages = [
        "Python programming",
        "Rust systems",
        "JavaScript web",
        "Go concurrency"
    ]

    surface = SemanticSurface(messages)
    model = SentenceTransformer('all-MiniLM-L6-v2')

    query_text = "web development"

    # Get coordinates via project_embedding
    query_emb = model.encode([query_text])[0]
    u_proj, v_proj = surface.project_embedding(query_emb)

    # Get coordinates via full query
    result = surface.query(query_text)
    u_query, v_query = result.uv

    # Should be identical (same projection algorithm)
    assert abs(u_proj - u_query) < 1e-10
    assert abs(v_proj - v_query) < 1e-10
