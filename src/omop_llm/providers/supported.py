"""The providers omop_llm supports, as explicit classes.

any-llm itself supports around fifty providers (see its own reference:
https://docs.mozilla.ai/any-llm/providers/).
omop_llm currently supports the following models: 
- local (``ollama``, ``llama-server`` via ``llamacpp``, ``vllm``), and
- cloud (``openai``, ``anthropic``, ``gemini``). 

Each class here subclasses both :class:`~omop_llm.providers.base.ProviderMixin`
(our contract: ``TOOL_USE``/``STRUCTURED_OUTPUT``, and the required
``canonical_model_name`` override) and any-llm's own provider class for
that provider.
"""

from __future__ import annotations

from typing import Any

import httpx
from any_llm.providers.anthropic.anthropic import AnthropicProvider as AnyLLMAnthropicProvider
from any_llm.providers.gemini.gemini import GeminiProvider as AnyLLMGeminiProvider
from any_llm.providers.llamacpp.llamacpp import LlamacppProvider as AnyLLMLlamacppProvider
from any_llm.providers.ollama.ollama import OllamaProvider as AnyLLMOllamaProvider
from any_llm.providers.openai.openai import OpenaiProvider as AnyLLMOpenaiProvider
from any_llm.providers.vllm.vllm import VllmProvider as AnyLLMVllmProvider
from any_llm.types.completion import CompletionParams
from oa_configurator import get_logger
from ollama import Options

from omop_llm.providers.base import ProviderMixin

_logger = get_logger(__name__)


class OllamaProvider(ProviderMixin, AnyLLMOllamaProvider):
    """Wrapped Ollama provider, for local dev and TRE fallback.
    Extends any-llm's own ``OllamaProvider`` with canonical model naming and
    embedding-dimension lookup via Ollama's native ``POST /api/show``.
    Supports structured output natively using ``response_format``.

    ``base_url`` defaults to ``http://localhost:11434`` if not given.
    ``api_key`` is not required.
    """

    TOOL_USE = True
    STRUCTURED_OUTPUT = True

    @classmethod
    def canonical_model_name(cls, name: str) -> str:
        """Require an explicit, immutable Ollama model tag.

        Rejects both untagged names and the mutable ``:latest`` tag: 
        ``:latest`` can silently repoint after an ``ollama pull``, breaking
        consistency between stored embeddings and new query embeddings.

        Parameters
        ----------
        name : str
            Model name with an explicit tag, e.g. ``"llama3:8b"`` or
            ``"nomic-embed-text:v1.5"``.

        Returns
        -------
        str
            The input name, validated and stripped of whitespace.

        Raises
        ------
        ValueError
            If the name has no tag, or if the tag is ``:latest``.
        """
        name = name.strip()
        if ":" not in name:
            raise ValueError(
                f"Ollama model name {name!r} must include an explicit tag. "
                f"Use a specific version (e.g. '{name}:8b') instead of relying on "
                "the mutable ':latest' pointer. Running 'ollama pull "
                f"{name}' can silently change which model version ':latest' "
                "refers to, breaking consistency between stored embeddings "
                "and new query embeddings."
            )

        _model_part, tag = name.rsplit(":", 1)
        if tag == "latest":
            raise ValueError(
                f"Ollama model name {name!r} uses the mutable ':latest' tag. "
                "':latest' can change between 'ollama pull' runs, breaking "
                "consistency between stored embeddings and new query "
                "embeddings. Use an explicit, immutable tag (e.g. "
                "'<model_name>:8b')."
            )
        return name

    def embedding_dimension_hint(self, model: str, *, api_base: str | None) -> int | None:
        """See :meth:`omop_llm.providers.base.ProviderMixin.embedding_dimension_hint`."""
        if api_base is None:
            return None
        response = httpx.post(f"{api_base.rstrip('/')}/api/show", json={"name": model}).json()
        return self._extract_embedding_length(response)

    async def async_embedding_dimension_hint(self, model: str, *, api_base: str | None) -> int | None:
        """See :meth:`omop_llm.providers.base.ProviderMixin.async_embedding_dimension_hint`."""
        if api_base is None:
            return None
        async with httpx.AsyncClient() as client:
            response = await client.post(f"{api_base.rstrip('/')}/api/show", json={"name": model})
        return self._extract_embedding_length(response.json())

    @staticmethod
    def _extract_embedding_length(response: dict) -> int | None:
        """Extract the embedding length from an Ollama ``/api/show`` response."""
        model_info = response.get("model_info", {})
        if not model_info:
            return None
        embedding_keys = [key for key in model_info if "embedding_length" in key]
        if len(embedding_keys) != 1:
            return None
        return int(model_info[embedding_keys[0]])

    @staticmethod
    def _convert_completion_params(params: CompletionParams, **kwargs: Any) -> dict[str, Any]:
        """Override any-llm's param conversion to fix/flag what Ollama's ``Options`` would silently drop.

        - ``max_tokens`` -> ``num_predict`` (Ollama's native name; ``num_predict``
          wins if both are present).
        - Any other key not in ``Options.model_fields`` or popped elsewhere
          is logged as a warning instead of silently vanishing.

        Notes
        -----
        - poppped_before_options: Popped in any_llm/providers/ollama/ollama.py:L.202-203
        - Tracked in: https://github.com/mozilla-ai/any-llm/issues/1206
        """
        popped_before_options = frozenset({"tools", "think"})
        converted = AnyLLMOllamaProvider._convert_completion_params(params, **kwargs)
        if "max_tokens" in converted:
            max_tokens = converted.pop("max_tokens")
            converted.setdefault("num_predict", max_tokens)
        unrecognized = converted.keys() - Options.model_fields.keys() - popped_before_options
        if unrecognized:
            _logger.warning(
                "Ollama will silently ignore unrecognized completion kwargs: %s",
                sorted(unrecognized),
            )
        return converted


class LlamacppProvider(ProviderMixin, AnyLLMLlamacppProvider):
    """llama.cpp's ``llama-server``. Covers local dev and a CUDA/TRE fallback profile.

    ``base_url`` defaults to ``http://127.0.0.1:8080/v1`` if not given.
    ``api_key`` is not required.
    """

    TOOL_USE = True
    STRUCTURED_OUTPUT = True

    @classmethod
    def canonical_model_name(cls, name: str) -> str:
        """No transformation: ``llama-server`` model names have no mutable-tag concern."""
        return name


class VllmProvider(ProviderMixin, AnyLLMVllmProvider):
    """vLLM, the preferred TRE/NVIDIA backend.

    ``base_url`` defaults to ``http://localhost:8000/v1`` if not given.
    ``api_key`` is optional since self-hosted vLLM commonly runs without auth.
    """

    TOOL_USE = True
    STRUCTURED_OUTPUT = True

    @classmethod
    def canonical_model_name(cls, name: str) -> str:
        """No transformation: vLLM model names have no mutable-tag concern."""
        return name


class OpenaiProvider(ProviderMixin, AnyLLMOpenaiProvider):
    """OpenAI, e.g. ``gpt-4o``.

    Defaults to ``https://api.openai.com/v1`` (any-llm's own explicit
    default) when ``base_url`` is not given, the real OpenAI API, same as
    leaving ``base_url`` unset in the ``openai`` SDK directly. Requires
    ``api_key`` (explicit, or the ``OPENAI_API_KEY`` environment
    variable); raises if neither is set.
    """

    TOOL_USE = True
    STRUCTURED_OUTPUT = True

    @classmethod
    def canonical_model_name(cls, name: str) -> str:
        """No transformation: OpenAI model names have no mutable-tag concern."""
        return name


class AnthropicProvider(ProviderMixin, AnyLLMAnthropicProvider):
    """Anthropic (Claude).

    ``base_url`` defaults to ``anthropic`` SDK default when not given.
    ``api_key`` is required (explicit, or the ``ANTHROPIC_API_KEY`` env var).

    Notes
    -----
    any-llm's ``get_provider_metadata()`` reports ``embedding=False``
    for Anthropic (it has no embeddings API), so
    :meth:`omop_llm.backend.ModelBackend.embed_texts` refuses this
    provider.
    """

    TOOL_USE = True
    STRUCTURED_OUTPUT = True

    @classmethod
    def canonical_model_name(cls, name: str) -> str:
        """No transformation: Anthropic model names have no mutable-tag concern."""
        return name


class GeminiProvider(ProviderMixin, AnyLLMGeminiProvider):
    """Gemini, e.g. ``gemini-2.5-pro``.

    ``base_url`` defaults to ``gemini`` SDK default when not given.
    ``api_key`` is required (explicit, or the ``GEMINI_API_KEY``/``GOOGLE_API_KEY`` env var).
    """

    TOOL_USE = True
    STRUCTURED_OUTPUT = True

    @classmethod
    def canonical_model_name(cls, name: str) -> str:
        """No transformation: Gemini model names have no mutable-tag concern."""
        return name
