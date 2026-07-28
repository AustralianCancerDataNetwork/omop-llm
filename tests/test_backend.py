"""ModelBackend: kwargs merging, response unpacking, capability gating, sync/async parity.

Uses ``FakeAnyLLMClient`` (see conftest.py) to test omop_llm's own wrapper
logic in isolation, plus real (offline-constructed, never called over the
network) provider instances to test ``build_backend``'s construction,
canonicalization, and capability-gate behavior.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from omop_llm.backend import ModelBackend, build_backend
from omop_llm.capabilities import ModelCapabilities
from omop_llm.errors import NoParsedOutputError, UnsupportedCapabilityError
from tests.conftest import (
    FakeAnyLLMClient,
    FakeChatCompletion,
    FakeChatCompletionMessage,
    FakeChoice,
    FakeEmbeddingItem,
    FakeEmbeddingResponse,
)

_CAPS = ModelCapabilities(
    streaming=True, embeddings=True, extended_thinking=True, tool_use=True, structured_output=True
)


class Answer(BaseModel):
    value: str


def _backend(fake_client: FakeAnyLLMClient, **kwargs) -> ModelBackend:
    return ModelBackend(_client=fake_client, model="m", capabilities=_CAPS, **kwargs)  # ty: ignore[invalid-argument-type]


@pytest.mark.parametrize("sync", [True, False])
async def test_complete_merges_configuration_and_call_kwargs(fake_client: FakeAnyLLMClient, sync: bool) -> None:
    fake_client.completion_response = FakeChatCompletion(
        choices=[FakeChoice(message=FakeChatCompletionMessage(content="hi"))]
    )
    backend = _backend(fake_client, configuration={"temperature": 0.0, "max_tokens": 8000})
    tools = [{"name": "lookup_item", "input_schema": {"type": "object"}}]

    if sync:
        backend.complete([{"role": "user", "content": "hi"}], tools=tools, max_tokens=2048)
    else:
        await backend.async_complete([{"role": "user", "content": "hi"}], tools=tools, max_tokens=2048)

    [call] = fake_client.completion_calls
    assert call["model"] == "m"
    assert call["tools"] == tools
    # explicit call-time max_tokens overrides the configured default
    assert call["max_tokens"] == 2048
    # configured temperature carries through untouched
    assert call["temperature"] == 0.0


@pytest.mark.parametrize("sync", [True, False])
async def test_complete_passes_response_format_through_untouched(fake_client: FakeAnyLLMClient, sync: bool) -> None:
    fake_client.completion_response = FakeChatCompletion(
        choices=[FakeChoice(message=FakeChatCompletionMessage(parsed={"ok": True}))]
    )
    backend = _backend(fake_client)

    class Dummy:
        pass

    if sync:
        backend.complete([{"role": "user", "content": "hi"}], response_format=Dummy)
    else:
        await backend.async_complete([{"role": "user", "content": "hi"}], response_format=Dummy)
    [call] = fake_client.completion_calls
    assert call["response_format"] is Dummy


@pytest.mark.parametrize("sync", [True, False])
async def test_embed_texts_unpacks_embedding_vectors(fake_client: FakeAnyLLMClient, sync: bool) -> None:
    fake_client.embedding_response = FakeEmbeddingResponse(
        data=[FakeEmbeddingItem(embedding=[0.1, 0.2]), FakeEmbeddingItem(embedding=[0.3, 0.4])]
    )
    backend = _backend(fake_client)
    backend.model = "embed-default"

    vectors = backend.embed_texts(["a", "b"]) if sync else await backend.async_embed_texts(["a", "b"])
    assert vectors == [[0.1, 0.2], [0.3, 0.4]]
    [call] = fake_client.embedding_calls
    assert call["model"] == "embed-default"
    assert call["inputs"] == ["a", "b"]


@pytest.mark.parametrize("sync", [True, False])
async def test_embed_texts_rejects_backend_without_embeddings(fake_client: FakeAnyLLMClient, sync: bool) -> None:
    no_embed_caps = ModelCapabilities(
        streaming=True, embeddings=False, extended_thinking=True, tool_use=True, structured_output=True
    )
    backend = ModelBackend(_client=fake_client, model="m", capabilities=no_embed_caps)  # ty: ignore[invalid-argument-type]
    with pytest.raises(UnsupportedCapabilityError):
        if sync:
            backend.embed_texts(["a"])
        else:
            await backend.async_embed_texts(["a"])


@pytest.mark.parametrize("sync", [True, False])
async def test_dimensions_prefers_configured_override(fake_client: FakeAnyLLMClient, sync: bool) -> None:
    backend = _backend(fake_client, configuration={"embedding_dim": 768})
    result = backend.dimensions() if sync else await backend.async_dimensions()
    assert result == 768
    assert fake_client.embedding_calls == []  # no live probe needed


@pytest.mark.parametrize("sync", [True, False])
async def test_dimensions_uses_provider_hint_before_live_probe(fake_client: FakeAnyLLMClient, sync: bool) -> None:
    fake_client.dimension_hint = 1024
    backend = _backend(fake_client)
    result = backend.dimensions() if sync else await backend.async_dimensions()
    assert result == 1024
    assert fake_client.embedding_calls == []  # hint short-circuits the live probe


@pytest.mark.parametrize("sync", [True, False])
async def test_dimensions_falls_back_to_live_probe(fake_client: FakeAnyLLMClient, sync: bool) -> None:
    fake_client.embedding_response = FakeEmbeddingResponse(data=[FakeEmbeddingItem(embedding=[0.0] * 384)])
    backend = _backend(fake_client)
    result = backend.dimensions() if sync else await backend.async_dimensions()
    assert result == 384


@pytest.mark.parametrize("sync", [True, False])
async def test_extract_rejects_backend_without_structured_output(fake_client: FakeAnyLLMClient, sync: bool) -> None:
    no_structured_caps = ModelCapabilities(
        streaming=True, embeddings=True, extended_thinking=True, tool_use=True, structured_output=False
    )
    backend = ModelBackend(_client=fake_client, model="m", capabilities=no_structured_caps)  # ty: ignore[invalid-argument-type]
    fake_client.completion_response = FakeChatCompletion(choices=[FakeChoice(message=FakeChatCompletionMessage())])
    with pytest.raises(UnsupportedCapabilityError):
        if sync:
            backend.extract([{"role": "user", "content": "hi"}], Answer)
        else:
            await backend.async_extract([{"role": "user", "content": "hi"}], Answer)


@pytest.mark.parametrize("sync", [True, False])
async def test_extract_unwraps_parsed_instance(fake_client: FakeAnyLLMClient, sync: bool) -> None:
    fake_client.completion_response = FakeChatCompletion(
        choices=[FakeChoice(message=FakeChatCompletionMessage(parsed=Answer(value="42")))]
    )
    backend = _backend(fake_client)
    result = (
        backend.extract([{"role": "user", "content": "hi"}], Answer)
        if sync
        else await backend.async_extract([{"role": "user", "content": "hi"}], Answer)
    )
    assert result == Answer(value="42")
    [call] = fake_client.completion_calls
    assert call["response_format"] is Answer


@pytest.mark.parametrize("sync", [True, False])
async def test_extract_raises_when_provider_did_not_honor_schema(fake_client: FakeAnyLLMClient, sync: bool) -> None:
    fake_client.completion_response = FakeChatCompletion(
        choices=[FakeChoice(message=FakeChatCompletionMessage(parsed=None))]
    )
    backend = _backend(fake_client)
    with pytest.raises(NoParsedOutputError):
        if sync:
            backend.extract([{"role": "user", "content": "hi"}], Answer)
        else:
            await backend.async_extract([{"role": "user", "content": "hi"}], Answer)


def test_build_backend_constructs_offline_for_local_provider() -> None:
    backend = build_backend(
        provider="llamacpp", model="local-chat", base_url="http://localhost:8080/v1"
    )
    assert backend.model == "local-chat"
    assert backend.capabilities.tool_use is True


def test_provider_property_reads_from_the_constructed_client_not_a_stored_field() -> None:
    backend = build_backend(provider="llamacpp", model="local-chat", base_url="http://localhost:8080/v1")
    assert backend.provider == "llamacpp"


def test_build_backend_passes_configuration_through() -> None:
    backend = build_backend(
        provider="llamacpp",
        model="local-chat",
        base_url="http://localhost:8080/v1",
        configuration={"temperature": 0.0},
    )
    assert backend.configuration == {"temperature": 0.0}


def test_build_backend_canonicalizes_the_model_name() -> None:
    backend = build_backend(provider="ollama", model="llama3:8b", base_url="http://localhost:11434")
    assert backend.model == "llama3:8b"


def test_build_backend_rejects_non_canonical_ollama_name() -> None:
    with pytest.raises(ValueError, match="explicit tag"):
        build_backend(provider="ollama", model="llama3", base_url="http://localhost:11434")


def test_build_backend_constructs_offline_for_embedding_capable_provider() -> None:
    backend = build_backend(
        provider="ollama", model="qwen3-embedding:0.6b", base_url="http://localhost:11434"
    )
    assert backend.model == "qwen3-embedding:0.6b"
    assert backend.capabilities.embeddings is True
