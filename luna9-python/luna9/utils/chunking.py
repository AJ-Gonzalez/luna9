"""
Text chunking utilities for document ingestion into semantic surfaces.

Provides multiple chunking strategies for breaking down large documents into
semantically meaningful units suitable for embedding and geometric memory storage.
"""

from typing import List, Dict, Any, Optional, Literal
import re
from dataclasses import dataclass


ChunkStrategy = Literal["paragraph", "sentence", "fixed", "semantic"]


@dataclass
class Chunk:
    """A text chunk with metadata."""

    text: str
    metadata: Dict[str, Any]
    start_pos: int
    end_pos: int

    def __len__(self) -> int:
        return len(self.text)


class TextChunker:
    """Chunks text documents into semantically meaningful units."""

    def __init__(
        self,
        strategy: ChunkStrategy = "paragraph",
        max_chunk_size: Optional[int] = None,
        overlap: int = 0,
        min_chunk_size: int = 10,
        preserve_structure: bool = True
    ):
        """
        Initialize text chunker.

        Args:
            strategy: Chunking strategy to use
                - "paragraph": Split on paragraph boundaries
                - "sentence": Split on sentence boundaries
                - "fixed": Fixed-size chunks with optional overlap
                - "semantic": Smart chunking preserving semantic units
            max_chunk_size: Maximum characters per chunk (None = unlimited)
            overlap: Number of characters to overlap between chunks
            min_chunk_size: Minimum characters per chunk (discard smaller)
            preserve_structure: Keep formatting markers like chapter headers
        """
        self.strategy = strategy
        self.max_chunk_size = max_chunk_size
        self.overlap = overlap
        self.min_chunk_size = min_chunk_size
        self.preserve_structure = preserve_structure

    def chunk(
        self,
        text: str,
        base_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Chunk]:
        """
        Chunk text according to strategy.

        Args:
            text: Text to chunk
            base_metadata: Metadata to include in all chunks

        Returns:
            List of Chunk objects with text and metadata
        """
        if base_metadata is None:
            base_metadata = {}

        if self.strategy == "paragraph":
            chunks = self._chunk_by_paragraph(text, base_metadata)
        elif self.strategy == "sentence":
            chunks = self._chunk_by_sentence(text, base_metadata)
        elif self.strategy == "fixed":
            chunks = self._chunk_fixed_size(text, base_metadata)
        elif self.strategy == "semantic":
            chunks = self._chunk_semantic(text, base_metadata)
        else:
            raise ValueError(f"Unknown chunking strategy: {self.strategy}")

        # Safety: if all chunks were filtered out, return the original text as a single chunk
        if not chunks and text.strip():
            chunks = [Chunk(
                text=text.strip(),
                metadata=base_metadata.copy(),
                start_pos=0,
                end_pos=len(text)
            )]

        return chunks

    def _chunk_by_paragraph(
        self,
        text: str,
        base_metadata: Dict[str, Any]
    ) -> List[Chunk]:
        """Split text on paragraph boundaries (double newline)."""
        # Split on double newlines, preserving chapter markers
        paragraphs = re.split(r'\n\s*\n', text)

        chunks = []
        position = 0

        for i, para in enumerate(paragraphs):
            para = para.strip()
            if len(para) < self.min_chunk_size:
                position += len(para) + 2  # Account for newlines
                continue

            # Detect structure markers
            is_chapter = bool(re.match(r'^(CHAPTER|Chapter|SECTION|Section)\s+\w+', para))

            metadata = {
                **base_metadata,
                "chunk_index": len(chunks),
                "chunk_type": "chapter_header" if is_chapter else "paragraph",
                "start_pos": position,
                "end_pos": position + len(para)
            }

            # Handle max chunk size
            if self.max_chunk_size and len(para) > self.max_chunk_size:
                # Split large paragraphs
                sub_chunks = self._split_large_chunk(para, position, metadata)
                chunks.extend(sub_chunks)
            else:
                chunks.append(Chunk(
                    text=para,
                    metadata=metadata,
                    start_pos=position,
                    end_pos=position + len(para)
                ))

            position += len(para) + 2

        return chunks

    def _chunk_by_sentence(
        self,
        text: str,
        base_metadata: Dict[str, Any]
    ) -> List[Chunk]:
        """Split text on sentence boundaries."""
        # Simple sentence splitting (could be improved with proper NLP)
        sentences = re.split(r'(?<=[.!?])\s+', text)

        chunks = []
        position = 0
        current_chunk = []
        current_size = 0

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            sentence_len = len(sentence)

            # Check if we need to flush current chunk
            if self.max_chunk_size and current_size + sentence_len > self.max_chunk_size:
                if current_chunk:
                    chunk_text = ' '.join(current_chunk)
                    chunks.append(Chunk(
                        text=chunk_text,
                        metadata={
                            **base_metadata,
                            "chunk_index": len(chunks),
                            "chunk_type": "sentence_group",
                            "sentence_count": len(current_chunk),
                            "start_pos": position - current_size,
                            "end_pos": position
                        },
                        start_pos=position - current_size,
                        end_pos=position
                    ))
                current_chunk = []
                current_size = 0

            current_chunk.append(sentence)
            current_size += sentence_len + 1  # +1 for space
            position += sentence_len + 1

        # Flush remaining
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            if len(chunk_text) >= self.min_chunk_size:
                chunks.append(Chunk(
                    text=chunk_text,
                    metadata={
                        **base_metadata,
                        "chunk_index": len(chunks),
                        "chunk_type": "sentence_group",
                        "sentence_count": len(current_chunk),
                        "start_pos": position - current_size,
                        "end_pos": position
                    },
                    start_pos=position - current_size,
                    end_pos=position
                ))

        return chunks

    def _chunk_fixed_size(
        self,
        text: str,
        base_metadata: Dict[str, Any]
    ) -> List[Chunk]:
        """Split text into fixed-size chunks with optional overlap."""
        if not self.max_chunk_size:
            raise ValueError("max_chunk_size required for fixed strategy")

        chunks = []
        position = 0

        while position < len(text):
            end = position + self.max_chunk_size
            chunk_text = text[position:end]

            if len(chunk_text) >= self.min_chunk_size:
                chunks.append(Chunk(
                    text=chunk_text,
                    metadata={
                        **base_metadata,
                        "chunk_index": len(chunks),
                        "chunk_type": "fixed",
                        "start_pos": position,
                        "end_pos": end
                    },
                    start_pos=position,
                    end_pos=end
                ))

            # Move position forward (accounting for overlap)
            position += self.max_chunk_size - self.overlap

        return chunks

    def _chunk_semantic(
        self,
        text: str,
        base_metadata: Dict[str, Any]
    ) -> List[Chunk]:
        """
        Smart chunking that preserves semantic units.

        Uses paragraph boundaries but tries to keep related paragraphs together
        based on heuristics (topic continuity, dialogue, etc.)
        """
        # Start with paragraph chunks
        para_chunks = self._chunk_by_paragraph(text, base_metadata)

        if not self.max_chunk_size:
            return para_chunks

        # Merge small consecutive chunks that seem related
        merged = []
        current_group = []
        current_size = 0

        for chunk in para_chunks:
            chunk_size = len(chunk.text)

            # Check if we should merge with current group
            can_merge = (
                current_size + chunk_size <= self.max_chunk_size
                and self._chunks_related(current_group, chunk)
            )

            if can_merge:
                current_group.append(chunk)
                current_size += chunk_size + 1  # +1 for joining space
            else:
                # Flush current group
                if current_group:
                    merged.append(self._merge_chunks(current_group, base_metadata))
                current_group = [chunk]
                current_size = chunk_size

        # Flush remaining
        if current_group:
            merged.append(self._merge_chunks(current_group, base_metadata))

        return merged

    def _split_large_chunk(
        self,
        text: str,
        start_pos: int,
        base_metadata: Dict[str, Any]
    ) -> List[Chunk]:
        """Split a chunk that exceeds max_chunk_size."""
        chunks = []
        position = 0

        while position < len(text):
            end = min(position + self.max_chunk_size, len(text))

            # Try to split on sentence boundary
            if end < len(text):
                last_period = text[position:end].rfind('. ')
                if last_period > self.max_chunk_size // 2:  # At least halfway
                    end = position + last_period + 1

            chunk_text = text[position:end].strip()
            if len(chunk_text) >= self.min_chunk_size:
                chunks.append(Chunk(
                    text=chunk_text,
                    metadata={
                        **base_metadata,
                        "chunk_index": len(chunks),
                        "split_from_large": True,
                        "start_pos": start_pos + position,
                        "end_pos": start_pos + end
                    },
                    start_pos=start_pos + position,
                    end_pos=start_pos + end
                ))

            position = end

        return chunks

    def _chunks_related(self, group: List[Chunk], new_chunk: Chunk) -> bool:
        """Heuristic to determine if chunks are semantically related."""
        if not group:
            return True

        last_chunk = group[-1]

        # Don't merge across chapter boundaries
        if (last_chunk.metadata.get("chunk_type") == "chapter_header" or
            new_chunk.metadata.get("chunk_type") == "chapter_header"):
            return False

        # Dialogue tends to stay together (multiple paragraphs with quotes)
        last_has_quotes = '"' in last_chunk.text or "'" in last_chunk.text
        new_has_quotes = '"' in new_chunk.text or "'" in new_chunk.text
        if last_has_quotes and new_has_quotes:
            return True

        # Short paragraphs often continue same topic
        if len(last_chunk.text) < 200 and len(new_chunk.text) < 200:
            return True

        return False

    def _merge_chunks(
        self,
        chunks: List[Chunk],
        base_metadata: Dict[str, Any]
    ) -> Chunk:
        """Merge multiple chunks into one."""
        if not chunks:
            raise ValueError("Cannot merge empty chunk list")

        if len(chunks) == 1:
            return chunks[0]

        merged_text = '\n\n'.join(c.text for c in chunks)

        return Chunk(
            text=merged_text,
            metadata={
                **base_metadata,
                "chunk_index": chunks[0].metadata["chunk_index"],
                "chunk_type": "merged",
                "merged_count": len(chunks),
                "start_pos": chunks[0].start_pos,
                "end_pos": chunks[-1].end_pos
            },
            start_pos=chunks[0].start_pos,
            end_pos=chunks[-1].end_pos
        )


def extract_chapters(text: str) -> List[Dict[str, Any]]:
    """
    Extract chapter information from text.

    Returns list of dicts with:
        - title: Chapter title
        - number: Chapter number (if found)
        - start_pos: Starting position
        - text: Chapter text
    """
    # Pattern for chapter headers
    chapter_pattern = r'^(CHAPTER|Chapter|SECTION|Section)\s+([\dIVXLCDM]+|One|Two|Three|[A-Z][a-z]+)\s*[:\-]?\s*(.*)$'

    chapters = []
    lines = text.split('\n')
    current_chapter = None
    current_text = []
    position = 0

    for line in lines:
        match = re.match(chapter_pattern, line.strip())

        if match:
            # Save previous chapter
            if current_chapter:
                current_chapter['text'] = '\n'.join(current_text).strip()
                chapters.append(current_chapter)

            # Start new chapter
            prefix, number, title = match.groups()
            current_chapter = {
                'title': title.strip() if title else f"{prefix} {number}",
                'number': number,
                'start_pos': position,
                'header': line.strip()
            }
            current_text = []
        else:
            if current_chapter is not None:
                current_text.append(line)

        position += len(line) + 1

    # Save last chapter
    if current_chapter:
        current_chapter['text'] = '\n'.join(current_text).strip()
        chapters.append(current_chapter)

    return chapters


def chunk_by_chapters(
    text: str,
    base_metadata: Optional[Dict[str, Any]] = None,
    chunk_chapters: bool = True,
    chunker: Optional[TextChunker] = None
) -> List[Chunk]:
    """
    Chunk text by chapters, optionally further chunking each chapter.

    Args:
        text: Full text to chunk
        base_metadata: Base metadata for all chunks
        chunk_chapters: If True, further chunk each chapter's text
        chunker: TextChunker to use for chapter content (default: paragraph)

    Returns:
        List of chunks with chapter metadata
    """
    if base_metadata is None:
        base_metadata = {}

    if chunker is None:
        chunker = TextChunker(strategy="paragraph", max_chunk_size=2000)

    chapters = extract_chapters(text)

    if not chapters:
        # No chapters found, just chunk the whole text
        return chunker.chunk(text, base_metadata)

    all_chunks = []

    for chapter in chapters:
        chapter_meta = {
            **base_metadata,
            "chapter_number": chapter['number'],
            "chapter_title": chapter['title'],
            "chapter_header": chapter['header']
        }

        if chunk_chapters:
            # Further chunk the chapter content
            chapter_chunks = chunker.chunk(chapter['text'], chapter_meta)
            all_chunks.extend(chapter_chunks)
        else:
            # Keep entire chapter as one chunk
            all_chunks.append(Chunk(
                text=chapter['text'],
                metadata={
                    **chapter_meta,
                    "chunk_index": len(all_chunks),
                    "chunk_type": "chapter",
                    "start_pos": chapter['start_pos']
                },
                start_pos=chapter['start_pos'],
                end_pos=chapter['start_pos'] + len(chapter['text'])
            ))

    return all_chunks
