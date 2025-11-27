"""
External integrations for Luna 9.

Provides connectors for various data sources including Project Gutenberg,
web scraping, and document format parsers, plus OpenRouter API for LLM access.
"""

from .gutenberg import (
    GutenbergText,
    fetch_gutenberg_text,
    load_gutenberg_text,
    get_domain_path_for_gutenberg,
    get_recommended_work_id,
    list_recommended_works,
    RECOMMENDED_WORKS
)

from .openrouter import (
    OpenRouterClient,
    ModelPresets,
    create_client,
)

__all__ = [
    # Gutenberg
    "GutenbergText",
    "fetch_gutenberg_text",
    "load_gutenberg_text",
    "get_domain_path_for_gutenberg",
    "get_recommended_work_id",
    "list_recommended_works",
    "RECOMMENDED_WORKS",
    # OpenRouter
    "OpenRouterClient",
    "ModelPresets",
    "create_client",
]
