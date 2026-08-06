"""Structured extraction fallback: the instructor allow-list guard rails.

No test here calls a real provider or instructor client over the network,
these test the guard rails (which providers are refused, and why)
:mod:`omop_llm.structured`'s module docstring explains.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from omop_llm.errors import UnsupportedCapabilityError
from omop_llm.providers import supported_providers
from omop_llm.structured import (
    _INSTRUCTOR_SAFE_PROVIDERS,
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
