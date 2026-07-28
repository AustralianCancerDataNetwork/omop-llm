"""Provider registry: the allow-list, capability declarations, and canonicalization dispatch."""

from __future__ import annotations

import pytest

from omop_llm.errors import UnsupportedProviderError
from omop_llm.providers import (
    canonical_model_name,
    capabilities_for,
    provider_class_for,
    supported_providers,
)


def test_registry_has_no_unexpected_providers() -> None:
    """Pins exact registry membership, so a stray ProviderMixin subclass (e.g. a test fixture) fails loudly instead of silently entering PROVIDER_REGISTRY."""
    assert supported_providers() == (
        "anthropic",
        "gemini",
        "llamacpp",
        "ollama",
        "openai",
        "vllm",
    )


def test_unregistered_provider_rejected() -> None:
    with pytest.raises(UnsupportedProviderError):
        provider_class_for("azure")


@pytest.mark.parametrize(
    ("provider", "expect_embeddings"),
    [
        ("ollama", True),
        ("llamacpp", True),
        ("vllm", True),
        ("openai", True),
        ("anthropic", False),  # Anthropic has no embeddings API
        ("gemini", True),
    ],
)
def test_capabilities_embeddings_match_any_llm_metadata(
    provider: str, expect_embeddings: bool
) -> None:
    caps = capabilities_for(provider)
    assert caps.embeddings is expect_embeddings


def test_capabilities_tool_use_and_structured_output_are_declared_not_inferred() -> None:
    for provider in supported_providers():
        caps = capabilities_for(provider)
        # These two are never read off any-llm's own metadata; it has no
        # such fields at all. Every registered provider currently declares
        # both True; this just pins that it comes from our own registry.
        assert caps.tool_use is True
        assert caps.structured_output is True


def test_unregistered_provider_capabilities_rejected() -> None:
    with pytest.raises(UnsupportedProviderError):
        capabilities_for("bedrock")


def test_canonical_model_name_dispatches_to_the_right_provider() -> None:
    # ollama has real transformation rules; every other registered provider
    # is currently a no-op passthrough.
    assert canonical_model_name("openai", "gpt-4o") == "gpt-4o"
    assert canonical_model_name("ollama", "llama3:8b") == "llama3:8b"


def test_canonical_model_name_unregistered_provider_rejected() -> None:
    with pytest.raises(UnsupportedProviderError):
        canonical_model_name("bedrock", "some-model")
