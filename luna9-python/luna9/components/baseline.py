"""
Baseline retrieval system using traditional cosine similarity.

This is the "traditional RAG" approach for comparison against the
semantic surface dual retrieval system.

Simple vector search: embed query, compute cosine similarity to all
message embeddings, return top-k most similar.
"""

from typing import List, Dict, Optional
from dataclasses import dataclass

import numpy as np
from sentence_transformers import SentenceTransformer


@dataclass
class BaselineResult:
    """Result from baseline cosine similarity retrieval."""
    similarities: List[tuple[int, float]]  # [(msg_idx, similarity), ...] sorted descending

    def get_messages(self, messages: List[str], k: int = 5) -> Dict:
        """
        Retrieve top-k messages by cosine similarity.

        Args:
            messages: Original message list
            k: Number of messages to return

        Returns:
            Dict with retrieved messages and metadata
        """
        return {
            'messages': [messages[idx] for idx, _ in self.similarities[:k]],
            'similarities': [sim for _, sim in self.similarities[:k]],
            'indices': [idx for idx, _ in self.similarities[:k]]
        }


class BaselineRetrieval:
    """
    Traditional RAG baseline using cosine similarity in embedding space.

    Provides same query interface as SemanticSurface for fair comparison.
    """

    def __init__(
        self,
        messages: List[str],
        embeddings: Optional[np.ndarray] = None,
        model_name: str = 'all-MiniLM-L6-v2'
    ):
        """
        Create baseline retrieval system from messages.

        Args:
            messages: List of text messages
            embeddings: Pre-computed embeddings (n, d) or None to compute
            model_name: sentence-transformers model to use if computing embeddings
        """
        self.messages = messages
        self.model_name = model_name

        # Embed messages if not provided
        if embeddings is None:
            print(f"Embedding {len(messages)} messages with {model_name}...")
            model = SentenceTransformer(model_name)
            embeddings = model.encode(messages, show_progress_bar=False)
            print(f"  Created embeddings: {embeddings.shape}")

        self.embeddings = embeddings
        self.embedding_dim = embeddings.shape[1]

        # Normalize embeddings for cosine similarity (so we can use dot product)
        self.normalized_embeddings = embeddings / np.linalg.norm(
            embeddings, axis=1, keepdims=True
        )

        print("Created baseline retrieval:")
        print(f"  Messages: {len(messages)}")
        print(f"  Embedding dim: {self.embedding_dim}")

    def query(self, query_text: str, k: int = 5) -> BaselineResult:
        """
        Query using cosine similarity.

        Args:
            query_text: Text query to search for
            k: Number of results to return

        Returns:
            BaselineResult with similarity-ranked messages
        """
        # Embed query
        model = SentenceTransformer(self.model_name)
        query_embedding = model.encode([query_text], show_progress_bar=False)[0]

        # Normalize query embedding
        query_normalized = query_embedding / np.linalg.norm(query_embedding)

        # Compute cosine similarity to all messages (dot product of normalized vectors)
        similarities = self.normalized_embeddings @ query_normalized

        # Create sorted list of (index, similarity)
        ranked = [
            (idx, float(sim))
            for idx, sim in enumerate(similarities)
        ]
        ranked.sort(key=lambda x: x[1], reverse=True)

        return BaselineResult(similarities=ranked)


def create_baseline_from_conversation(
    messages: List[str],
    model_name: str = 'all-MiniLM-L6-v2'
) -> BaselineRetrieval:
    """
    Convenience function to create baseline retrieval from conversation.

    Args:
        messages: List of messages
        model_name: Embedding model to use

    Returns:
        BaselineRetrieval ready for querying
    """
    return BaselineRetrieval(messages, model_name=model_name)
