"""Tests for text chunking utilities."""

import pytest
from luna9.utils.chunking import (
    TextChunker,
    Chunk,
    extract_chapters,
    chunk_by_chapters
)


# Sample texts for testing
SAMPLE_PARAGRAPH_TEXT = """
This is the first paragraph. It contains multiple sentences. They are simple.

This is the second paragraph. It's also straightforward. Multiple sentences here too.

This is the third paragraph.
""".strip()

SAMPLE_CHAPTER_TEXT = """
CHAPTER I - The Beginning

This is the first chapter. It has some content here.
More content in the first chapter.

CHAPTER II - The Middle

This is the second chapter. Different content now.
The story continues here.

Chapter III: The End

Final chapter content here.
""".strip()


class TestTextChunker:
    """Tests for TextChunker class."""

    def test_paragraph_chunking(self):
        """Test chunking by paragraphs."""
        chunker = TextChunker(strategy="paragraph")
        chunks = chunker.chunk(SAMPLE_PARAGRAPH_TEXT)

        assert len(chunks) == 3
        assert all(isinstance(c, Chunk) for c in chunks)
        assert "first paragraph" in chunks[0].text
        assert "second paragraph" in chunks[1].text
        assert "third paragraph" in chunks[2].text

    def test_paragraph_with_max_size(self):
        """Test paragraph chunking with max size constraint."""
        chunker = TextChunker(strategy="paragraph", max_chunk_size=100)
        long_text = "A" * 500  # Long paragraph
        chunks = chunker.chunk(long_text)

        # Should split the long paragraph
        assert len(chunks) > 1
        assert all(len(c.text) <= 100 for c in chunks)

    def test_sentence_chunking(self):
        """Test chunking by sentences."""
        text = "First sentence. Second sentence! Third sentence? Fourth sentence."
        chunker = TextChunker(strategy="sentence", max_chunk_size=50)
        chunks = chunker.chunk(text)

        assert len(chunks) > 0
        for chunk in chunks:
            assert chunk.metadata['chunk_type'] == 'sentence_group'

    def test_fixed_size_chunking(self):
        """Test fixed-size chunking."""
        text = "A" * 1000
        chunker = TextChunker(strategy="fixed", max_chunk_size=100, overlap=10)
        chunks = chunker.chunk(text)

        assert len(chunks) > 0
        # Check overlaps
        for i in range(len(chunks) - 1):
            # Chunks should overlap by 10 chars
            end_of_current = chunks[i].end_pos
            start_of_next = chunks[i + 1].start_pos
            assert end_of_current - start_of_next == 10

    def test_semantic_chunking(self):
        """Test semantic chunking strategy."""
        chunker = TextChunker(strategy="semantic", max_chunk_size=200)
        chunks = chunker.chunk(SAMPLE_PARAGRAPH_TEXT)

        assert len(chunks) > 0
        # Should preserve semantic boundaries (paragraphs)
        for chunk in chunks:
            assert '\n\n' not in chunk.text or chunk.metadata.get('merged_count')

    def test_min_chunk_size(self):
        """Test minimum chunk size filtering."""
        text = "Short.\n\nThis is longer and should be kept."
        chunker = TextChunker(strategy="paragraph", min_chunk_size=20)
        chunks = chunker.chunk(text)

        # Short paragraph should be filtered out
        assert len(chunks) == 1
        assert "longer" in chunks[0].text

    def test_metadata_preservation(self):
        """Test that base metadata is preserved in chunks."""
        base_meta = {"author": "Test Author", "title": "Test Title"}
        chunker = TextChunker(strategy="paragraph")
        chunks = chunker.chunk(SAMPLE_PARAGRAPH_TEXT, base_metadata=base_meta)

        for chunk in chunks:
            assert chunk.metadata['author'] == "Test Author"
            assert chunk.metadata['title'] == "Test Title"
            assert 'chunk_index' in chunk.metadata

    def test_chunk_positions(self):
        """Test that chunk positions are tracked correctly."""
        chunker = TextChunker(strategy="paragraph")
        chunks = chunker.chunk(SAMPLE_PARAGRAPH_TEXT)

        # Positions should be sequential
        for i, chunk in enumerate(chunks):
            assert chunk.start_pos >= 0
            assert chunk.end_pos > chunk.start_pos
            if i > 0:
                # Current chunk should start after or near previous chunk end
                assert chunk.start_pos >= chunks[i - 1].start_pos


class TestChapterExtraction:
    """Tests for chapter extraction."""

    def test_extract_chapters(self):
        """Test extracting chapters from text."""
        chapters = extract_chapters(SAMPLE_CHAPTER_TEXT)

        assert len(chapters) == 3
        assert chapters[0]['number'] == 'I'
        assert chapters[1]['number'] == 'II'
        assert chapters[2]['number'] == 'III'

        assert "The Beginning" in chapters[0]['title']
        assert "first chapter" in chapters[0]['text']

    def test_chapter_with_no_chapters(self):
        """Test extraction when no chapters present."""
        text = "Just some plain text without chapters."
        chapters = extract_chapters(text)

        assert len(chapters) == 0

    def test_various_chapter_formats(self):
        """Test different chapter header formats."""
        text = """
CHAPTER 1

Content here.

Chapter Two: A Title

More content.

SECTION III - Another Way

Final content.
        """.strip()

        chapters = extract_chapters(text)

        assert len(chapters) == 3
        assert chapters[0]['number'] == '1'
        assert chapters[1]['number'] == 'Two'
        assert chapters[2]['number'] == 'III'


class TestChunkByChapters:
    """Tests for chunk_by_chapters function."""

    def test_chunk_by_chapters(self):
        """Test chunking text by chapters."""
        chunks = chunk_by_chapters(
            SAMPLE_CHAPTER_TEXT,
            chunk_chapters=False
        )

        # Should have one chunk per chapter
        assert len(chunks) == 3
        assert all('chapter_number' in c.metadata for c in chunks)

    def test_chunk_chapters_content(self):
        """Test further chunking of chapter content."""
        # Create chunker that will split each chapter
        chunker = TextChunker(strategy="paragraph", max_chunk_size=50)

        chunks = chunk_by_chapters(
            SAMPLE_CHAPTER_TEXT,
            chunk_chapters=True,
            chunker=chunker
        )

        # Should have multiple chunks (chapters split into paragraphs)
        assert len(chunks) > 3
        # All should have chapter metadata
        assert all('chapter_number' in c.metadata for c in chunks)

    def test_no_chapters_fallback(self):
        """Test that it falls back to regular chunking when no chapters."""
        text = "Plain text without any chapter markers."
        chunks = chunk_by_chapters(text, chunk_chapters=True)

        assert len(chunks) > 0
        # Should not have chapter metadata
        assert 'chapter_number' not in chunks[0].metadata


class TestChunkDataclass:
    """Tests for Chunk dataclass."""

    def test_chunk_creation(self):
        """Test creating a Chunk."""
        chunk = Chunk(
            text="Test text",
            metadata={"key": "value"},
            start_pos=0,
            end_pos=9
        )

        assert chunk.text == "Test text"
        assert chunk.metadata['key'] == "value"
        assert len(chunk) == 9

    def test_chunk_len(self):
        """Test Chunk length."""
        chunk = Chunk(
            text="Hello, world!",
            metadata={},
            start_pos=0,
            end_pos=13
        )

        assert len(chunk) == 13
