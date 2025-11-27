"""
Project Gutenberg corpus loader for parametric surfaces benchmark.

Downloads public domain texts and chunks them intelligently using Luna9's
smart chunker (respects sentence boundaries, no mid-sentence crimes).
"""

import requests
from pathlib import Path
from typing import List, Optional
import re

# Luna9 smart chunker
from luna9.utils.chunking import TextChunker


class GutenbergLoader:
    """Load and chunk Project Gutenberg texts."""

    # Public domain books with good variety
    BOOKS = [
        (1342, "Pride and Prejudice", "Jane Austen"),
        (11, "Alice's Adventures in Wonderland", "Lewis Carroll"),
        (84, "Frankenstein", "Mary Shelley"),
        (1661, "The Adventures of Sherlock Holmes", "Arthur Conan Doyle"),
        (174, "The Picture of Dorian Gray", "Oscar Wilde"),
        (2701, "Moby Dick", "Herman Melville"),
        (1952, "The Yellow Wallpaper", "Charlotte Perkins Gilman"),
        (1260, "Jane Eyre", "Charlotte Brontë"),
        (16, "Peter Pan", "J. M. Barrie"),
        (2814, "Dubliners", "James Joyce"),
    ]

    def __init__(self, cache_dir: Path = Path("whitepapers/gutenberg_cache")):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def download_book(self, book_id: int) -> Optional[str]:
        """Download a book from Project Gutenberg."""
        cache_path = self.cache_dir / f"{book_id}.txt"

        # Check cache first
        if cache_path.exists():
            print(f"  Loading from cache: {cache_path.name}")
            return cache_path.read_text(encoding='utf-8')

        # Download from Gutenberg
        url = f"https://www.gutenberg.org/files/{book_id}/{book_id}-0.txt"
        alt_url = f"https://www.gutenberg.org/files/{book_id}/{book_id}.txt"

        print(f"  Downloading book {book_id}...")

        try:
            response = requests.get(url, timeout=30)
            if response.status_code != 200:
                # Try alternative URL
                response = requests.get(alt_url, timeout=30)

            if response.status_code == 200:
                text = response.text
                # Cache it
                cache_path.write_text(text, encoding='utf-8')
                return text
            else:
                print(f"  ✗ Failed to download book {book_id}: {response.status_code}")
                return None

        except Exception as e:
            print(f"  ✗ Error downloading book {book_id}: {e}")
            return None

    def clean_gutenberg_text(self, text: str) -> str:
        """
        Remove Gutenberg header/footer boilerplate.

        Gutenberg books have standard headers and footers with license info.
        We want just the actual book content.
        """
        # Find the start of the actual content
        # Usually marked by "*** START OF"
        start_markers = [
            "*** START OF THIS PROJECT GUTENBERG EBOOK",
            "*** START OF THE PROJECT GUTENBERG EBOOK",
            "***START OF THE PROJECT GUTENBERG EBOOK"
        ]

        start_idx = 0
        for marker in start_markers:
            idx = text.find(marker)
            if idx != -1:
                # Find the end of this line
                start_idx = text.find('\n', idx) + 1
                break

        # Find the end of the actual content
        # Usually marked by "*** END OF"
        end_markers = [
            "*** END OF THIS PROJECT GUTENBERG EBOOK",
            "*** END OF THE PROJECT GUTENBERG EBOOK",
            "***END OF THE PROJECT GUTENBERG EBOOK"
        ]

        end_idx = len(text)
        for marker in end_markers:
            idx = text.find(marker)
            if idx != -1:
                end_idx = idx
                break

        # Extract just the book content
        cleaned = text[start_idx:end_idx].strip()

        return cleaned

    def load_corpus(
        self,
        num_books: int = 5,
        target_chunks: int = 1000,
        chunk_size: int = 500
    ) -> List[str]:
        """
        Load and chunk Project Gutenberg books.

        Args:
            num_books: Number of books to load
            target_chunks: Target number of chunks (approximate)
            chunk_size: Target chunk size in characters

        Returns:
            List of text chunks
        """
        chunks = []

        # Load books
        for i, (book_id, title, author) in enumerate(self.BOOKS[:num_books]):
            print(f"\n[{i+1}/{num_books}] {title} by {author}")

            text = self.download_book(book_id)
            if not text:
                continue

            # Clean Gutenberg boilerplate
            text = self.clean_gutenberg_text(text)

            # Chunk using Luna9's smart chunker
            print(f"  Chunking (target: {chunk_size} chars)...")
            chunker = TextChunker(
                strategy='smart',
                target_size=chunk_size,
                max_chunk_size=chunk_size * 2,  # Allow up to 2x if needed
                overlap_sentences=1  # Small overlap for context
            )
            chunk_objects = chunker.chunk(text)
            book_chunks = [c.text for c in chunk_objects]  # Extract just the text

            print(f"  Generated {len(book_chunks)} chunks")
            chunks.extend(book_chunks)  # chunks is List[str], accumulating across books

            # Stop if we have enough chunks
            if len(chunks) >= target_chunks:
                print(f"\n[OK] Reached target of {target_chunks} chunks")
                break

        print(f"\n[OK] Loaded {len(chunks)} total chunks from {i+1} books")
        return chunks[:target_chunks]  # Trim to exact target


def create_ground_truth_queries(corpus: List[str]) -> List[dict]:
    """
    Create hand-labeled queries for evaluation.

    Strategy: Search corpus for semantically rich passages and create
    natural queries for them.
    """
    queries = []

    # Helper to get text from either string or Chunk
    def get_text(item):
        return item.text if hasattr(item, 'text') else item

    # Find passages containing common English words/themes
    search_terms = [
        ('marriage', 'thematic'),
        ('love', 'thematic'),
        ('family', 'thematic'),
        ('money', 'thematic'),
        ('happiness', 'thematic'),
        ('friend', 'thematic'),
        ('house', 'factual'),
        ('letter', 'factual'),
        ('visit', 'factual'),
        ('conversation', 'thematic'),
    ]

    for search_term, query_type in search_terms:
        # Find passages containing this term
        matching_indices = []
        for idx, passage in enumerate(corpus):
            text = get_text(passage).lower()
            if search_term in text:
                matching_indices.append(idx)

        if matching_indices:
            # Create a query using the search term
            queries.append({
                'query': f"passages about {search_term}",
                'relevant_indices': matching_indices[:5],  # Top 5 as ground truth
                'query_type': query_type
            })

            # Stop if we have enough queries
            if len(queries) >= 20:
                break

    return queries


if __name__ == '__main__':
    """Test the loader."""
    loader = GutenbergLoader()

    # Load small test corpus
    corpus = loader.load_corpus(
        num_books=2,
        target_chunks=100,
        chunk_size=500
    )

    print(f"\nSample chunk:\n{str(corpus[0])[:200]}...")
    print(f"Type: {type(corpus[0])}")

    # Create queries
    queries = create_ground_truth_queries(corpus)
    print(f"\nGenerated {len(queries)} queries")
    if queries:
        print(f"Example: {queries[0]['query']}")
