"""ModelBackend: kwargs merging, response unpacking, capability gating, sync/async parity.

Uses ``FakeAnyLLMClient`` (see conftest.py) to test omop_llm's own wrapper
logic in isolation, plus real (offline-constructed, never called over the
network) provider instances to test ``build_model_backend``'s construction,
canonicalization, and capability-gate behavior.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from omop_llm.backend import ModelBackend, build_model_backend
from omop_llm.capabilities import ModelCapabilities
from omop_llm.embeddings import EmbeddingRole
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
async def test_embed_texts_batch_size_chunks_and_preserves_order(fake_client: FakeAnyLLMClient, sync: bool) -> None:
    texts = ["a", "bb", "ccc", "dddd", "e"]

    def fake_embedding(**kwargs) -> FakeEmbeddingResponse:
        fake_client.embedding_calls.append(kwargs)
        return FakeEmbeddingResponse(data=[FakeEmbeddingItem(embedding=[float(len(t))]) for t in kwargs["inputs"]])

    async def fake_aembedding(**kwargs) -> FakeEmbeddingResponse:
        return fake_embedding(**kwargs)

    # Instance attribute assignment doesn't need a `self` parameter, unlike
    # the class-declared bound method ty checks this against.
    fake_client._embedding = fake_embedding  # ty: ignore[invalid-assignment]
    fake_client.aembedding = fake_aembedding  # ty: ignore[invalid-assignment]
    backend = _backend(fake_client)

    vectors = (
        backend.embed_texts(texts, batch_size=2)
        if sync
        else await backend.async_embed_texts(texts, batch_size=2)
    )
    assert vectors == [[1.0], [2.0], [3.0], [4.0], [1.0]]
    assert [len(call["inputs"]) for call in fake_client.embedding_calls] == [2, 2, 1]


@pytest.mark.parametrize("sync", [True, False])
async def test_embed_texts_rejects_non_positive_batch_size(fake_client: FakeAnyLLMClient, sync: bool) -> None:
    backend = _backend(fake_client)
    with pytest.raises(ValueError, match="positive"):
        if sync:
            backend.embed_texts(["a"], batch_size=0)
        else:
            await backend.async_embed_texts(["a"], batch_size=0)


@pytest.mark.parametrize("sync", [True, False])
async def test_embed_texts_applies_role_prefix(fake_client: FakeAnyLLMClient, sync: bool) -> None:
    fake_client.embedding_response = FakeEmbeddingResponse(data=[FakeEmbeddingItem(embedding=[0.1])])
    backend = _backend(
        fake_client, configuration={"document_prefix": "passage: ", "query_prefix": "query: "}
    )

    if sync:
        backend.embed_texts(["diabetes"], role=EmbeddingRole.DOCUMENT)
    else:
        await backend.async_embed_texts(["diabetes"], role=EmbeddingRole.DOCUMENT)
    [call] = fake_client.embedding_calls
    assert call["inputs"] == ["passage: diabetes"]


@pytest.mark.parametrize("sync", [True, False])
async def test_embed_texts_query_role_uses_query_prefix(fake_client: FakeAnyLLMClient, sync: bool) -> None:
    fake_client.embedding_response = FakeEmbeddingResponse(data=[FakeEmbeddingItem(embedding=[0.1])])
    backend = _backend(
        fake_client, configuration={"document_prefix": "passage: ", "query_prefix": "query: "}
    )

    if sync:
        backend.embed_texts(["hypertension"], role=EmbeddingRole.QUERY)
    else:
        await backend.async_embed_texts(["hypertension"], role=EmbeddingRole.QUERY)
    [call] = fake_client.embedding_calls
    assert call["inputs"] == ["query: hypertension"]


@pytest.mark.parametrize("sync", [True, False])
async def test_embed_texts_no_role_leaves_text_untouched(fake_client: FakeAnyLLMClient, sync: bool) -> None:
    fake_client.embedding_response = FakeEmbeddingResponse(data=[FakeEmbeddingItem(embedding=[0.1])])
    backend = _backend(fake_client, configuration={"document_prefix": "passage: "})

    if sync:
        backend.embed_texts(["diabetes"])
    else:
        await backend.async_embed_texts(["diabetes"])
    [call] = fake_client.embedding_calls
    assert call["inputs"] == ["diabetes"]


@pytest.mark.parametrize("sync", [True, False])
async def test_embed_texts_role_with_no_configured_prefix_is_a_noop(
    fake_client: FakeAnyLLMClient, sync: bool
) -> None:
    fake_client.embedding_response = FakeEmbeddingResponse(data=[FakeEmbeddingItem(embedding=[0.1])])
    backend = _backend(fake_client)

    if sync:
        backend.embed_texts(["diabetes"], role=EmbeddingRole.DOCUMENT)
    else:
        await backend.async_embed_texts(["diabetes"], role=EmbeddingRole.DOCUMENT)
    [call] = fake_client.embedding_calls
    assert call["inputs"] == ["diabetes"]


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


@pytest.mark.parametrize("sync", [True, False])
async def test_extract_raises_after_exhausting_max_retries(fake_client: FakeAnyLLMClient, sync: bool) -> None:
    fake_client.completion_response = FakeChatCompletion(
        choices=[FakeChoice(message=FakeChatCompletionMessage(parsed=None))]
    )
    backend = _backend(fake_client)
    with pytest.raises(NoParsedOutputError):
        if sync:
            backend.extract([{"role": "user", "content": "hi"}], Answer, max_retries=2)
        else:
            await backend.async_extract([{"role": "user", "content": "hi"}], Answer, max_retries=2)
    assert len(fake_client.completion_calls) == 3  # initial attempt + 2 retries


@pytest.mark.parametrize("sync", [True, False])
async def test_extract_retries_after_no_parsed_output_and_succeeds(
    fake_client: FakeAnyLLMClient, sync: bool
) -> None:
    responses = [
        FakeChatCompletion(choices=[FakeChoice(message=FakeChatCompletionMessage(parsed=None))]),
        FakeChatCompletion(choices=[FakeChoice(message=FakeChatCompletionMessage(parsed=Answer(value="42")))]),
    ]
    calls: list[dict] = []

    def fake_completion(**kwargs) -> FakeChatCompletion:
        calls.append(kwargs)
        return responses[len(calls) - 1]

    async def fake_acompletion(**kwargs) -> FakeChatCompletion:
        return fake_completion(**kwargs)

    # Instance attribute assignment doesn't need a `self` parameter, unlike
    # the class-declared bound method ty checks this against.
    fake_client.completion = fake_completion  # ty: ignore[invalid-assignment]
    fake_client.acompletion = fake_acompletion  # ty: ignore[invalid-assignment]
    backend = _backend(fake_client)

    result = (
        backend.extract([{"role": "user", "content": "hi"}], Answer, max_retries=1)
        if sync
        else await backend.async_extract([{"role": "user", "content": "hi"}], Answer, max_retries=1)
    )
    assert result == Answer(value="42")
    assert len(calls) == 2
    # the retried call carries the original message plus a corrective follow-up
    assert calls[1]["messages"][0] == {"role": "user", "content": "hi"}
    assert len(calls[1]["messages"]) == 2
    assert calls[1]["messages"][1]["role"] == "user"


@pytest.mark.parametrize("sync", [True, False])
async def test_extract_retries_after_validation_error_and_succeeds(
    fake_client: FakeAnyLLMClient, sync: bool
) -> None:
    fake_client.completion_response = FakeChatCompletion(
        choices=[FakeChoice(message=FakeChatCompletionMessage(parsed=Answer(value="42")))]
    )
    calls: list[dict] = []

    def fake_completion(**kwargs) -> FakeChatCompletion:
        calls.append(kwargs)
        if len(calls) == 1:
            Answer.model_validate({})  # raises pydantic.ValidationError: 'value' is required
        assert fake_client.completion_response is not None
        return fake_client.completion_response

    async def fake_acompletion(**kwargs) -> FakeChatCompletion:
        return fake_completion(**kwargs)

    # Instance attribute assignment doesn't need a `self` parameter, unlike
    # the class-declared bound method ty checks this against.
    fake_client.completion = fake_completion  # ty: ignore[invalid-assignment]
    fake_client.acompletion = fake_acompletion  # ty: ignore[invalid-assignment]
    backend = _backend(fake_client)

    result = (
        backend.extract([{"role": "user", "content": "hi"}], Answer, max_retries=1)
        if sync
        else await backend.async_extract([{"role": "user", "content": "hi"}], Answer, max_retries=1)
    )
    assert result == Answer(value="42")
    assert len(calls) == 2


def test_build_backend_constructs_offline_for_local_provider() -> None:
    backend = build_model_backend(
        provider="llamacpp", model="local-chat", base_url="http://localhost:8080/v1"
    )
    assert backend.model == "local-chat"
    assert backend.capabilities.tool_use is True


def test_provider_property_reads_from_the_constructed_client_not_a_stored_field() -> None:
    backend = build_model_backend(provider="llamacpp", model="local-chat", base_url="http://localhost:8080/v1")
    assert backend.provider == "llamacpp"


def test_build_backend_passes_configuration_through() -> None:
    backend = build_model_backend(
        provider="llamacpp",
        model="local-chat",
        base_url="http://localhost:8080/v1",
        configuration={"temperature": 0.0},
    )
    assert backend.configuration == {"temperature": 0.0}


def test_build_backend_canonicalizes_the_model_name() -> None:
    backend = build_model_backend(provider="ollama", model="llama3:8b", base_url="http://localhost:11434")
    assert backend.model == "llama3:8b"


def test_build_backend_rejects_non_canonical_ollama_name() -> None:
    with pytest.raises(ValueError, match="explicit tag"):
        build_model_backend(provider="ollama", model="llama3", base_url="http://localhost:11434")


def test_build_backend_constructs_offline_for_embedding_capable_provider() -> None:
    backend = build_model_backend(
        provider="ollama", model="qwen3-embedding:0.6b", base_url="http://localhost:11434"
    )
    assert backend.model == "qwen3-embedding:0.6b"
    assert backend.capabilities.embeddings is True


def test_build_backend_warns_on_missing_prefixes_for_embedding_model(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level("WARNING", logger="omop_llm.embeddings"):
        build_model_backend(provider="ollama", model="qwen3-embedding:0.6b", base_url="http://localhost:11434")
    assert "document_prefix" in caplog.text
    assert "query_prefix" in caplog.text


def test_build_backend_no_warning_when_prefixes_configured(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level("WARNING", logger="omop_llm.embeddings"):
        build_model_backend(
            provider="ollama",
            model="qwen3-embedding:0.6b",
            base_url="http://localhost:11434",
            configuration={"document_prefix": "search_document: ", "query_prefix": "search_query: "},
        )
    assert caplog.text == ""


def test_build_backend_no_prefix_warning_for_non_embedding_provider(caplog: pytest.LogCaptureFixture) -> None:
    # anthropic is the one provider in the registry with embeddings=False.
    with caplog.at_level("WARNING", logger="omop_llm.embeddings"):
        build_model_backend(provider="anthropic", model="claude-haiku-4-5", api_key="sk-test")
    assert caplog.text == ""
