"""
Project Gutenberg integration for Luna 9.

Fetches and processes public domain texts from Project Gutenberg
for ingestion into semantic memory surfaces.
"""

from typing import Optional, Dict, Any, Tuple
import re
import urllib.request
import urllib.error
from pathlib import Path


class GutenbergText:
    """Represents a Project Gutenberg text with metadata."""

    def __init__(
        self,
        text: str,
        title: str,
        author: Optional[str] = None,
        gutenberg_id: Optional[int] = None,
        language: str = "en",
        release_date: Optional[str] = None
    ):
        self.text = text
        self.title = title
        self.author = author
        self.gutenberg_id = gutenberg_id
        self.language = language
        self.release_date = release_date

    @property
    def metadata(self) -> Dict[str, Any]:
        """Get metadata dict suitable for Luna 9 domains."""
        meta = {
            "source_type": "gutenberg",
            "title": self.title,
            "language": self.language
        }

        if self.author:
            meta["author"] = self.author
        if self.gutenberg_id:
            meta["gutenberg_id"] = self.gutenberg_id
        if self.release_date:
            meta["release_date"] = self.release_date

        return meta

    def __repr__(self) -> str:
        author_str = f" by {self.author}" if self.author else ""
        return f"<GutenbergText: {self.title}{author_str}>"


def fetch_gutenberg_text(
    gutenberg_id: int,
    mirror: str = "https://www.gutenberg.org"
) -> GutenbergText:
    """
    Fetch text from Project Gutenberg by ID.

    Args:
        gutenberg_id: Project Gutenberg work ID
        mirror: Base URL for Gutenberg mirror

    Returns:
        GutenbergText object with content and metadata

    Raises:
        urllib.error.URLError: If fetch fails
        ValueError: If text cannot be parsed
    """
    # Construct URL (PG uses directory structure based on ID)
    # e.g., ID 84 -> /files/84/84-0.txt
    url = f"{mirror}/files/{gutenberg_id}/{gutenberg_id}-0.txt"

    try:
        with urllib.request.urlopen(url) as response:
            raw_text = response.read().decode('utf-8')
    except urllib.error.URLError as e:
        # Try alternate URL structure
        url = f"{mirror}/cache/epub/{gutenberg_id}/pg{gutenberg_id}.txt"
        try:
            with urllib.request.urlopen(url) as response:
                raw_text = response.read().decode('utf-8')
        except urllib.error.URLError:
            raise ValueError(f"Could not fetch Gutenberg ID {gutenberg_id}") from e

    # Parse metadata and clean text
    text, metadata = _parse_gutenberg_text(raw_text, gutenberg_id)

    return GutenbergText(
        text=text,
        gutenberg_id=gutenberg_id,
        **metadata
    )


def load_gutenberg_text(file_path: str) -> GutenbergText:
    """
    Load a locally saved Project Gutenberg text file.

    Args:
        file_path: Path to .txt file

    Returns:
        GutenbergText object
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(path, 'r', encoding='utf-8') as f:
        raw_text = f.read()

    # Try to extract Gutenberg ID from filename or content
    gutenberg_id = None
    if match := re.search(r'pg(\d+)', path.name):
        gutenberg_id = int(match.group(1))

    text, metadata = _parse_gutenberg_text(raw_text, gutenberg_id)

    return GutenbergText(
        text=text,
        gutenberg_id=gutenberg_id,
        **metadata
    )


def _parse_gutenberg_text(
    raw_text: str,
    gutenberg_id: Optional[int] = None
) -> Tuple[str, Dict[str, Any]]:
    """
    Parse Project Gutenberg text, extracting metadata and cleaning content.

    Args:
        raw_text: Raw text from Gutenberg file
        gutenberg_id: Optional Gutenberg ID

    Returns:
        Tuple of (cleaned_text, metadata_dict)
    """
    metadata = {}

    # Extract title
    if title_match := re.search(r'Title:\s*(.+?)(?:\r?\n)', raw_text):
        metadata['title'] = title_match.group(1).strip()
    else:
        metadata['title'] = f"Gutenberg {gutenberg_id}" if gutenberg_id else "Unknown"

    # Extract author
    if author_match := re.search(r'Author:\s*(.+?)(?:\r?\n)', raw_text):
        metadata['author'] = author_match.group(1).strip()

    # Extract language
    if lang_match := re.search(r'Language:\s*(.+?)(?:\r?\n)', raw_text):
        metadata['language'] = lang_match.group(1).strip()
    else:
        metadata['language'] = 'en'

    # Extract release date
    if date_match := re.search(r'Release Date:\s*(.+?)(?:\r?\n)', raw_text):
        metadata['release_date'] = date_match.group(1).strip()

    # Remove Gutenberg header and footer
    text = _remove_gutenberg_boilerplate(raw_text)

    return text, metadata


def _remove_gutenberg_boilerplate(text: str) -> str:
    """
    Remove Project Gutenberg header and footer boilerplate.

    PG texts have standard headers/footers with licensing info.
    """
    # Find start marker
    start_markers = [
        r'\*\*\* START OF THIS PROJECT GUTENBERG',
        r'\*\*\*START OF THE PROJECT GUTENBERG',
        r'START OF THE PROJECT GUTENBERG EBOOK'
    ]

    start_pos = 0
    for marker in start_markers:
        if match := re.search(marker, text, re.IGNORECASE):
            # Start after this line
            next_line = text.find('\n', match.end())
            if next_line != -1:
                start_pos = next_line + 1
            break

    # Find end marker
    end_markers = [
        r'\*\*\* END OF THIS PROJECT GUTENBERG',
        r'\*\*\*END OF THE PROJECT GUTENBERG',
        r'END OF THE PROJECT GUTENBERG EBOOK'
    ]

    end_pos = len(text)
    for marker in end_markers:
        if match := re.search(marker, text, re.IGNORECASE):
            end_pos = match.start()
            break

    # Extract content
    content = text[start_pos:end_pos].strip()

    # Remove excessive whitespace
    content = re.sub(r'\n\s*\n\s*\n+', '\n\n', content)  # Max 2 newlines
    content = re.sub(r'[ \t]+', ' ', content)  # Normalize spaces

    return content


def get_domain_path_for_gutenberg(
    work: GutenbergText,
    base_path: str = "foundation/literature"
) -> str:
    """
    Generate appropriate domain path for a Gutenberg work.

    Args:
        work: GutenbergText object
        base_path: Base path for literature domains

    Returns:
        Domain path like "foundation/literature/frankenstein"
    """
    # Sanitize title for use in path
    title_slug = work.title.lower()
    title_slug = re.sub(r'[^\w\s-]', '', title_slug)  # Remove punctuation
    title_slug = re.sub(r'[-\s]+', '_', title_slug)  # Replace spaces/hyphens with underscore
    title_slug = title_slug[:50]  # Limit length

    return f"{base_path}/{title_slug}"


# Common public domain works for testing/demos
RECOMMENDED_WORKS = {
    "frankenstein": 84,
    "alice_in_wonderland": 11,
    "pride_and_prejudice": 1342,
    "dracula": 345,
    "sherlock_holmes": 1661,  # Adventures of Sherlock Holmes
    "metamorphosis": 5200,  # Kafka
    "moby_dick": 2701,
    "great_gatsby": 64317,
    "picture_of_dorian_gray": 174,
    "time_machine": 35,  # H.G. Wells
    "jekyll_and_hyde": 43,
    "yellow_wallpaper": 1952,  # Charlotte Perkins Gilman
    "the_raven": 17192,  # Poe poetry collection
    "sense_and_sensibility": 161,
    "tale_of_two_cities": 98,
}


def get_recommended_work_id(slug: str) -> Optional[int]:
    """
    Get Gutenberg ID for a recommended work by slug.

    Args:
        slug: Work identifier (e.g., "frankenstein")

    Returns:
        Gutenberg ID or None if not found
    """
    return RECOMMENDED_WORKS.get(slug.lower())


def list_recommended_works() -> Dict[str, int]:
    """Get dict of recommended public domain works."""
    return RECOMMENDED_WORKS.copy()
