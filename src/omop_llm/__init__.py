from .interface import EmbeddingClient, LLMClient, InstructorClient, CHAT_MESSAGE_DICT, LLMClientError
from .providers import (
    EmbeddingProvider,
    OllamaProvider,
    OpenAICompatProvider,
    get_provider_for_api_base,
)

__all__ = [
    # Primary names
    "EmbeddingClient",
    "InstructorClient",
    "CHAT_MESSAGE_DICT",
    "LLMClientError",
    # Providers
    "EmbeddingProvider",
    "OllamaProvider",
    "OpenAICompatProvider",
    "get_provider_for_api_base",
    # Backward-compatibility alias
    "LLMClient",
]