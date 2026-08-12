"""OllamaProvider's real, provider-specific behavior.

Ported from ``omop-emb/src/omop_emb/embeddings/embedding_providers.py``'s
``OllamaProvider``. ``httpx`` calls are monkeypatched, no test here touches
the network.
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest
from any_llm.types.completion import CompletionParams

from omop_llm.providers.supported import OllamaProvider


@pytest.mark.parametrize("name", ["llama3:8b", "nomic-embed-text:v1.5"])
def test_canonical_model_name_accepts_explicit_tags(name: str) -> None:
    assert OllamaProvider.canonical_model_name(name) == name


def test_canonical_model_name_rejects_untagged_names() -> None:
    with pytest.raises(ValueError, match="explicit tag"):
        OllamaProvider.canonical_model_name("llama3")


def test_canonical_model_name_rejects_mutable_latest_tag() -> None:
    with pytest.raises(ValueError, match="latest"):
        OllamaProvider.canonical_model_name("llama3:latest")


def test_canonical_model_name_is_idempotent() -> None:
    once = OllamaProvider.canonical_model_name("llama3:8b")
    twice = OllamaProvider.canonical_model_name(once)
    assert once == twice


def test_embedding_dimension_hint_returns_none_without_api_base() -> None:
    provider = OllamaProvider(api_key=None, api_base="http://localhost:11434")
    assert provider.embedding_dimension_hint("nomic-embed-text:v1.5", api_base=None) is None


def test_embedding_dimension_hint_parses_api_show_response(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_post(url: str, json: dict[str, Any]) -> httpx.Response:
        assert url == "http://localhost:11434/api/show"
        assert json == {"name": "nomic-embed-text:v1.5"}
        return httpx.Response(200, json={"model_info": {"nomic-embed-text.embedding_length": 768}})

    monkeypatch.setattr(httpx, "post", fake_post)
    provider = OllamaProvider(api_key=None, api_base="http://localhost:11434")
    result = provider.embedding_dimension_hint("nomic-embed-text:v1.5", api_base="http://localhost:11434")
    assert result == 768


def test_embedding_dimension_hint_returns_none_when_metadata_is_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_post(url: str, json: dict[str, Any]) -> httpx.Response:
        return httpx.Response(200, json={"model_info": {}})

    monkeypatch.setattr(httpx, "post", fake_post)
    provider = OllamaProvider(api_key=None, api_base="http://localhost:11434")
    result = provider.embedding_dimension_hint("some-model:8b", api_base="http://localhost:11434")
    assert result is None


async def test_async_embedding_dimension_hint_returns_none_without_api_base() -> None:
    provider = OllamaProvider(api_key=None, api_base="http://localhost:11434")
    result = await provider.async_embedding_dimension_hint("nomic-embed-text:v1.5", api_base=None)
    assert result is None


async def test_async_embedding_dimension_hint_parses_api_show_response(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_post(self: httpx.AsyncClient, url: str, json: dict[str, Any]) -> httpx.Response:
        assert url == "http://localhost:11434/api/show"
        return httpx.Response(200, json={"model_info": {"nomic-embed-text.embedding_length": 768}})

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)
    provider = OllamaProvider(api_key=None, api_base="http://localhost:11434")
    result = await provider.async_embedding_dimension_hint(
        "nomic-embed-text:v1.5", api_base="http://localhost:11434"
    )
    assert result == 768


def test_convert_completion_params_translates_max_tokens_to_num_predict() -> None:
    params = CompletionParams(model_id="llama3:8b", messages=[{"role": "user", "content": "hi"}], max_tokens=512)
    converted = OllamaProvider._convert_completion_params(params)
    assert "max_tokens" not in converted
    assert converted["num_predict"] == 512


def test_convert_completion_params_prefers_explicit_num_predict_over_max_tokens() -> None:
    params = CompletionParams(model_id="llama3:8b", messages=[{"role": "user", "content": "hi"}], max_tokens=512)
    converted = OllamaProvider._convert_completion_params(params, num_predict=128)
    assert converted["num_predict"] == 128
    assert "max_tokens" not in converted


def test_convert_completion_params_omits_num_predict_when_max_tokens_not_set() -> None:
    params = CompletionParams(model_id="llama3:8b", messages=[{"role": "user", "content": "hi"}])
    converted = OllamaProvider._convert_completion_params(params)
    assert "num_predict" not in converted


def test_convert_completion_params_warns_on_unrecognized_kwargs(caplog: pytest.LogCaptureFixture) -> None:
    params = CompletionParams(model_id="llama3:8b", messages=[{"role": "user", "content": "hi"}])
    with caplog.at_level("WARNING", logger="omop_llm.providers.supported"):
        converted = OllamaProvider._convert_completion_params(params, bogus_kwarg="x")
    assert converted["bogus_kwarg"] == "x"
    assert "bogus_kwarg" in caplog.text


def test_convert_completion_params_does_not_warn_on_tools_or_think(caplog: pytest.LogCaptureFixture) -> None:
    params = CompletionParams(model_id="llama3:8b", messages=[{"role": "user", "content": "hi"}])
    with caplog.at_level("WARNING", logger="omop_llm.providers.supported"):
        OllamaProvider._convert_completion_params(params, tools=[{"type": "function"}], think=True)
    assert caplog.text == ""
