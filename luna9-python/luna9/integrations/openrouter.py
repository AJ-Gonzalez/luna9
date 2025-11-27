"""
OpenRouter API Client

Provides interface to OpenRouter for testing initiative architecture with
small, fast models (Haiku, GPT-4o Mini, Llama 3.1 8B).
"""

import os
from typing import Optional, Dict, Any, List
import requests


class OpenRouterClient:
    """
    Client for OpenRouter API.

    Supports simple text completion for testing initiative emergence.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://openrouter.ai/api/v1"
    ):
        """
        Initialize OpenRouter client.

        Args:
            api_key: OpenRouter API key (defaults to OPENROUTER_API_KEY env var)
            base_url: Base URL for OpenRouter API

        Example:
            Using environment variable (.env file):
            ```
            # .env
            OPENROUTER_API_KEY=sk-or-v1-xxxxx

            # Python
            from dotenv import load_dotenv
            from luna9 import create_client

            load_dotenv()
            client = create_client()
            ```

            Or pass directly:
            ```python
            from luna9 import OpenRouterClient

            client = OpenRouterClient(api_key="sk-or-v1-xxxxx")
            ```

        Raises:
            ValueError: If no API key is provided via argument or environment
        """
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenRouter API key required. Set OPENROUTER_API_KEY env var "
                "or pass api_key to constructor."
            )

        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        })

    def complete(
        self,
        prompt: str,
        model: str = "anthropic/claude-3-haiku",
        max_tokens: int = 500,
        temperature: float = 0.7,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Get completion from OpenRouter model.

        Args:
            prompt: Input prompt
            model: Model identifier (e.g., "anthropic/claude-3-haiku")
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional API parameters

        Returns:
            Response dict with completion and metadata
        """
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            **kwargs
        }

        response = self.session.post(
            f"{self.base_url}/chat/completions",
            json=payload
        )
        response.raise_for_status()

        data = response.json()

        # Extract completion text
        completion = data["choices"][0]["message"]["content"]

        return {
            "completion": completion,
            "model": model,
            "usage": data.get("usage", {}),
            "raw_response": data
        }

    def complete_simple(
        self,
        prompt: str,
        model: str = "anthropic/claude-3-haiku"
    ) -> str:
        """
        Get just the completion text (convenience method).

        Args:
            prompt: Input prompt
            model: Model identifier

        Returns:
            Completion text
        """
        result = self.complete(prompt, model=model)
        return result["completion"]

    def complete_conversation(
        self,
        messages: List[Dict[str, str]],
        model: str = "anthropic/claude-3-haiku",
        max_tokens: int = 500,
        temperature: float = 0.7
    ) -> str:
        """
        Get completion with full conversation history.

        Args:
            messages: List of message dicts with 'role' and 'content'
            model: Model identifier
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            Completion text
        """
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature
        }

        response = self.session.post(
            f"{self.base_url}/chat/completions",
            json=payload
        )
        response.raise_for_status()

        data = response.json()
        return data["choices"][0]["message"]["content"]

    def list_models(self) -> List[Dict[str, Any]]:
        """
        List all available models from OpenRouter.

        Returns:
            List of model dicts with id, name, pricing, context length, etc.
        """
        response = self.session.get(f"{self.base_url}/models")
        response.raise_for_status()
        data = response.json()
        return data.get("data", [])

    def get_model_info(self, model_id: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed info for a specific model.

        Args:
            model_id: Model identifier (e.g., "anthropic/claude-3-haiku")

        Returns:
            Model info dict or None if not found
        """
        models = self.list_models()
        for model in models:
            if model.get("id") == model_id:
                return model
        return None

    def is_model_available(self, model_id: str) -> bool:
        """
        Check if a model is currently available.

        Args:
            model_id: Model identifier

        Returns:
            True if model exists and is available
        """
        info = self.get_model_info(model_id)
        return info is not None

    def filter_models(
        self,
        free_only: bool = False,
        max_price_per_1m: Optional[float] = None,
        min_context: Optional[int] = None,
        provider: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Filter available models by criteria.

        Args:
            free_only: Only return free models
            max_price_per_1m: Maximum price per 1M tokens (prompt + completion avg)
            min_context: Minimum context window size
            provider: Filter by provider (e.g., "anthropic", "openai")

        Returns:
            Filtered list of model dicts
        """
        models = self.list_models()
        filtered = []

        for model in models:
            # Check free
            if free_only:
                pricing = model.get("pricing", {})
                prompt_price = float(pricing.get("prompt", "1"))
                completion_price = float(pricing.get("completion", "1"))
                if prompt_price > 0 or completion_price > 0:
                    continue

            # Check max price
            if max_price_per_1m is not None:
                pricing = model.get("pricing", {})
                prompt_price = float(pricing.get("prompt", "999"))
                completion_price = float(pricing.get("completion", "999"))
                avg_price = (prompt_price + completion_price) / 2
                # Prices are per token, convert to per 1M
                avg_price_per_1m = avg_price * 1_000_000
                if avg_price_per_1m > max_price_per_1m:
                    continue

            # Check context length
            if min_context is not None:
                context_length = model.get("context_length", 0)
                if context_length < min_context:
                    continue

            # Check provider
            if provider is not None:
                model_id = model.get("id", "")
                if not model_id.startswith(f"{provider}/"):
                    continue

            filtered.append(model)

        return filtered


class ModelPresets:
    """
    Preset model configurations for testing.

    These are small, fast models suitable for initiative testing.
    """

    # Fast and cheap models for iteration
    HAIKU = "anthropic/claude-haiku-4.5"
    GPT4O_MINI = "openai/gpt-4o-mini"
    DEEPSEEK = "deepseek/deepseek-chat"
    MISTRAL = "mistralai/mistral-nemo"

    # Medium models for comparison
    SONNET = "anthropic/claude-sonnet-4.5"
    GPT4O = "openai/gpt-4o"

    @classmethod
    def all_test_models(cls) -> List[str]:
        """Return list of all fast test models."""
        return [cls.HAIKU, cls.GPT4O_MINI, cls.DEEPSEEK, cls.MISTRAL]

    @classmethod
    def all_models(cls) -> List[str]:
        """Return list of all available models."""
        return [
            cls.HAIKU,
            cls.GPT4O_MINI,
            cls.DEEPSEEK,
            cls.MISTRAL,
            cls.SONNET,
            cls.GPT4O
        ]


def create_client(api_key: Optional[str] = None) -> OpenRouterClient:
    """
    Factory function for OpenRouter client.

    Args:
        api_key: Optional API key (uses env var if not provided)

    Returns:
        Configured OpenRouter client
    """
    return OpenRouterClient(api_key=api_key)
