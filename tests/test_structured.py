"""Structured extraction fallback: the instructor allow-list guard rails.

No test here calls a real provider or instructor client over the network,
these test the guard rails (which providers are refused, and why)
:mod:`omop_llm.structured`'s module docstring explains.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import BaseModel

from omop_llm.errors import UnsupportedCapabilityError
from omop_llm.providers import supported_providers
from omop_llm.structured import (
    _INSTRUCTOR_SAFE_PROVIDERS,
    _LOCAL_PLACEHOLDER_API_KEY,
    _instructor_client_kwargs,
    async_extract_with_retry,
    extract_with_retry,
)


class Answer(BaseModel):
    value: str



_UNSAFE_PROVIDERS = sorted(set(supported_providers()) - _INSTRUCTOR_SAFE_PROVIDERS) + ["azure"]

def test_instructor_safe_providers_is_the_openai_compat_native_set() -> None:
    # Deliberately excludes ollama (instructor's own Ollama builder uses the
    # OpenAI-compat shim, not native /api/chat, see structured.py's module
    # docstring) and anthropic/gemini (not vouched for here at all).
    assert _INSTRUCTOR_SAFE_PROVIDERS == frozenset({"openai", "llamacpp", "vllm"})


@pytest.mark.parametrize("provider", _UNSAFE_PROVIDERS)
def test_extract_with_retry_rejects_unsafe_providers(provider: str) -> None:
    with pytest.raises(UnsupportedCapabilityError):
        extract_with_retry(provider, "some-model", [{"role": "user", "content": "hi"}], Answer, base_url="http://x")


@pytest.mark.parametrize("provider", _UNSAFE_PROVIDERS)
async def test_async_extract_with_retry_rejects_unsafe_providers(provider: str) -> None:
    with pytest.raises(UnsupportedCapabilityError):
        await async_extract_with_retry(
            provider, "some-model", [{"role": "user", "content": "hi"}], Answer, base_url="http://x"
        )


def test_extract_with_retry_requires_base_url_for_self_hosted_providers() -> None:
    with pytest.raises(ValueError, match="base_url"):
        extract_with_retry("llamacpp", "local-chat", [{"role": "user", "content": "hi"}], Answer)


async def test_async_extract_with_retry_requires_base_url_for_self_hosted_providers() -> None:
    with pytest.raises(ValueError, match="base_url"):
        await async_extract_with_retry("llamacpp", "local-chat", [{"role": "user", "content": "hi"}], Answer)


class TestInstructorClientKwargs:
    """llamacpp/vllm don't require real credentials, but the openai SDK
    client instructor builds underneath still requires a non-empty
    api_key string to construct at all -- this is what previously made
    extract_with_retry("llamacpp", ...) fail with no api_key given."""

    @pytest.mark.parametrize("provider", ["llamacpp", "vllm"])
    def test_local_provider_without_api_key_gets_placeholder(self, provider: str) -> None:
        kwargs = _instructor_client_kwargs(provider, base_url="http://x", api_key=None, async_client=False)
        assert kwargs["api_key"] == _LOCAL_PLACEHOLDER_API_KEY

    def test_openai_without_api_key_gets_no_placeholder(self) -> None:
        # Real OpenAI must keep falling through to OPENAI_API_KEY, not a fake key.
        kwargs = _instructor_client_kwargs("openai", base_url=None, api_key=None, async_client=False)
        assert "api_key" not in kwargs

    def test_explicit_api_key_is_never_overridden(self) -> None:
        kwargs = _instructor_client_kwargs("llamacpp", base_url="http://x", api_key="real-key", async_client=False)
        assert kwargs["api_key"] == "real-key"

    def test_async_client_flag_is_forwarded(self) -> None:
        kwargs = _instructor_client_kwargs("openai", base_url=None, api_key="k", async_client=True)
        assert kwargs["async_client"] is True


class TestInstructorClientConstruction:
    """End-to-end: instructor.from_provider is mocked, but extract_with_retry's
    own kwargs-building and call plumbing run for real."""

    def test_extract_with_retry_passes_placeholder_key_for_local_provider(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import instructor

        fake_client = MagicMock()
        fake_client.chat.completions.create.return_value = Answer(value="ok")
        captured: dict = {}

        def fake_from_provider(model_string: str, **kwargs):
            captured.update(kwargs)
            return fake_client

        monkeypatch.setattr(instructor, "from_provider", fake_from_provider)

        result = extract_with_retry(
            "llamacpp", "local-chat", [{"role": "user", "content": "hi"}], Answer, base_url="http://x"
        )
        assert result == Answer(value="ok")
        assert captured["api_key"] == _LOCAL_PLACEHOLDER_API_KEY
        assert captured["async_client"] is False

    async def test_async_extract_with_retry_passes_placeholder_key_for_local_provider(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import instructor

        fake_client = MagicMock()
        fake_client.chat.completions.create = AsyncMock(return_value=Answer(value="ok"))
        captured: dict = {}

        def fake_from_provider(model_string: str, **kwargs):
            captured.update(kwargs)
            return fake_client

        monkeypatch.setattr(instructor, "from_provider", fake_from_provider)

        result = await async_extract_with_retry(
            "vllm", "local-chat", [{"role": "user", "content": "hi"}], Answer, base_url="http://x"
        )
        assert result == Answer(value="ok")
        assert captured["api_key"] == _LOCAL_PLACEHOLDER_API_KEY
        assert captured["async_client"] is True

    def test_extract_with_retry_never_fakes_a_key_for_real_openai(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import instructor

        fake_client = MagicMock()
        fake_client.chat.completions.create.return_value = Answer(value="ok")
        captured: dict = {}

        def fake_from_provider(model_string: str, **kwargs):
            captured.update(kwargs)
            return fake_client

        monkeypatch.setattr(instructor, "from_provider", fake_from_provider)

        extract_with_retry("openai", "gpt-4o", [{"role": "user", "content": "hi"}], Answer, api_key="sk-real")
        assert captured["api_key"] == "sk-real"
