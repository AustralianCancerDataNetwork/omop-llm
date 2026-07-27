"""Shared test fixtures.

Contract tests here exercise omop_llm's own wrapper logic (kwargs merging,
response unpacking, capability gating, provider registration) against
either a fake ``AnyLLM`` client or the real any-llm provider classes
constructed offline (no network call happens at construction time, only
``.completion()``/``.acompletion()``/``._embedding()``/``.aembedding()``
touch the network, and no test here calls those on a real provider).
Nothing in this suite requires a running Ollama/llama-server/vLLM instance
or a live API key.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from omop_llm.providers.base import ProviderMixin


@dataclass
class FakeChatCompletionMessage:
    content: str | None = None
    parsed: Any = None


@dataclass
class FakeChoice:
    message: FakeChatCompletionMessage


@dataclass
class FakeChatCompletion:
    choices: list[FakeChoice]


@dataclass
class FakeEmbeddingItem:
    embedding: list[float]


@dataclass
class FakeEmbeddingResponse:
    data: list[FakeEmbeddingItem]


@dataclass
class FakeAnyLLMClient(ProviderMixin):
    """Stands in for a constructed any-llm provider instance.

    Records every call it receives so tests can assert on exactly what
    :class:`omop_llm.backend.ModelBackend` passed through, without needing
    a real provider or network access. Method names match the real
    ``AnyLLM`` surface :class:`~omop_llm.backend.ModelBackend` calls:
    ``completion``/``acompletion`` for chat, ``_embedding``/``aembedding``
    for embeddings (matching any-llm's own asymmetric naming, confirmed by
    reading ``any_llm/api.py`` and ``any_llm/any_llm.py`` directly), and
    ``embedding_dimension_hint``/``async_embedding_dimension_hint`` for the
    provider-specific dimension fast path.

    Subclasses :class:`~omop_llm.providers.base.ProviderMixin`, not just
    ``AnyLLM``'s duck-typed surface: every real ``_client`` a
    :class:`~omop_llm.backend.ModelBackend` is ever built with also is one,
    since :data:`~omop_llm.providers.registry.PROVIDER_REGISTRY` only
    contains classes that are both. Not doing so here would make this fake
    a less accurate stand-in than the objects it replaces.
    """

    TOOL_USE = True
    STRUCTURED_OUTPUT = True

    completion_response: FakeChatCompletion | None = None
    embedding_response: FakeEmbeddingResponse | None = None
    dimension_hint: int | None = None
    completion_calls: list[dict[str, Any]] = field(default_factory=list)
    embedding_calls: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def canonical_model_name(cls, name: str) -> str:
        return name

    def completion(self, **kwargs: Any) -> FakeChatCompletion:
        self.completion_calls.append(kwargs)
        assert self.completion_response is not None
        return self.completion_response

    async def acompletion(self, **kwargs: Any) -> FakeChatCompletion:
        self.completion_calls.append(kwargs)
        assert self.completion_response is not None
        return self.completion_response

    def _embedding(self, **kwargs: Any) -> FakeEmbeddingResponse:
        self.embedding_calls.append(kwargs)
        assert self.embedding_response is not None
        return self.embedding_response

    async def aembedding(self, **kwargs: Any) -> FakeEmbeddingResponse:
        self.embedding_calls.append(kwargs)
        assert self.embedding_response is not None
        return self.embedding_response

    def embedding_dimension_hint(self, model: str, *, api_base: str | None) -> int | None:
        return self.dimension_hint

    async def async_embedding_dimension_hint(self, model: str, *, api_base: str | None) -> int | None:
        return self.dimension_hint


@pytest.fixture
def fake_client() -> FakeAnyLLMClient:
    return FakeAnyLLMClient()
