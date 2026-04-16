"""Embedding provider abstractions for omop-llm.

An EmbeddingProvider encapsulates the two things that vary across embedding
backends:

- **Model name canonicalisation** — e.g. Ollama requires a tag such as
  ``llama3:latest`` while OpenAI-style names carry no tags.
- **Embedding dimension retrieval** — Ollama exposes a ``/api/show`` endpoint;
  OpenAI-compatible APIs do not have an equivalent.

The provider is inferred automatically from the ``api_base`` URL via
:func:`get_provider_for_api_base`, but can also be supplied explicitly to
:class:`~omop_llm.interface.client.LLMClient` for custom or future backends.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from httpx import URL

import requests


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers.

    Subclass this to support a new embedding backend. The two abstract methods
    capture the only behaviour that differs between providers at the
    ``LLMClient`` level; everything else (batched embedding calls via the
    OpenAI-compatible ``/v1/embeddings`` endpoint, similarity helpers, etc.)
    is shared and lives in ``LLMClient`` directly.
    """

    @abstractmethod
    def canonical_model_name(self, name: str) -> str:
        """Return the canonical form of *name* for this provider.

        The canonical form is the identifier used as a stable key in the
        embedding registry (``omop-emb``) and passed verbatim to the API.
        Implementations should be idempotent — calling this on an already-
        canonical name must return the same string unchanged.

        Parameters
        ----------
        name : str
            Raw model name as supplied by the caller, e.g. ``'llama3'`` or
            ``'text-embedding-3-small'``.

        Returns
        -------
        str
            Canonical model name, e.g. ``'llama3:latest'`` or
            ``'text-embedding-3-small'``.
        """
        ...

    @abstractmethod
    def get_embedding_dim(self, model: str, api_base: URL) -> int:
        """Return the embedding dimension for *model* served at *api_base*.

        Parameters
        ----------
        model : str
            Canonical model name (already processed by
            :meth:`canonical_model_name`).
        api_base : URL
            Base URL of the API endpoint, e.g.
            ``'http://localhost:11434/v1'``.

        Returns
        -------
        int
            Number of dimensions in the embedding vector.

        Raises
        ------
        ValueError
            If the dimension cannot be determined from the API response.
        NotImplementedError
            If the provider does not support automatic dimension retrieval.
        """
        ...


class OllamaProvider(EmbeddingProvider):
    """Provider for models served by Ollama.

    Canonical model names always include a tag (``name:tag``).  If no tag is
    present ``':latest'`` is appended.  Embedding dimensions are retrieved via
    Ollama's ``POST /api/show`` endpoint.
    """

    def canonical_model_name(self, name: str) -> str:
        """Append ``':latest'`` if *name* carries no tag.

        Parameters
        ----------
        name : str
            Model name, e.g. ``'llama3'`` or ``'llama3:8b'``.

        Returns
        -------
        str
            Model name with tag, e.g. ``'llama3:latest'`` or ``'llama3:8b'``.
        """
        name = name.strip()
        if ":" not in name:
            name = f"{name}:latest"
        return name

    def get_embedding_dim(self, model: str, api_base: URL) -> int:
        """Query ``POST /api/show`` for the embedding dimension.

        Parameters
        ----------
        model : str
            Canonical model name (with tag).
        api_base : URL
            Ollama API base URL, e.g. ``'http://localhost:11434/v1'``.

        Returns
        -------
        int
            Embedding vector dimension.

        Raises
        ------
        ValueError
            If model info or the embedding length key is absent from the
            Ollama response.
        """
        base_url = str(api_base).replace("/v1", "").rstrip("/")
        response = requests.post(f"{base_url}/api/show", json={"name": model}).json()
        model_info = response.get("model_info", {})

        embedding_keys = [k for k in model_info if "embedding_length" in k]
        if len(embedding_keys) == 1:
            return int(model_info[embedding_keys[0]])

        raise ValueError(
            f"Could not determine embedding dimension from Ollama response for "
            f"model '{model}'. Response: {response}"
        )


class OpenAICompatProvider(EmbeddingProvider):
    """Provider for OpenAI-compatible APIs (OpenAI, Azure OpenAI, etc.).

    Model names require no tag normalisation.  Embedding dimensions are not
    available via a query endpoint, so :meth:`get_embedding_dim` raises
    :exc:`NotImplementedError` — pass the dimension explicitly via the
    ``embedding_dim`` parameter of :class:`~omop_llm.interface.client.LLMClient`
    instead.
    """

    def canonical_model_name(self, name: str) -> str:
        """Return *name* unchanged (no tag normalisation required).

        Parameters
        ----------
        name : str
            Model name, e.g. ``'text-embedding-3-small'``.

        Returns
        -------
        str
            The same model name, stripped of surrounding whitespace.
        """
        return name.strip()

    def get_embedding_dim(self, model: str, api_base: str) -> int:
        raise NotImplementedError(
            f"Automatic embedding dimension retrieval is not supported for "
            f"OpenAI-compatible endpoints (api_base='{api_base}'). "
            f"Pass the dimension explicitly via the 'embedding_dim' parameter "
            f"when constructing LLMClient."
        )


def get_provider_for_api_base(
    api_base: str,
    api_key: str = "ollama",
) -> EmbeddingProvider:
    """Infer the appropriate :class:`EmbeddingProvider` from *api_base*.

    Detection rules (evaluated in order):

    1. ``'ollama'`` appears anywhere in *api_base* → :class:`OllamaProvider`
    2. *api_base* is a localhost/loopback URL **and** *api_key* is ``'ollama'``
       → :class:`OllamaProvider`
    3. All other URLs → :class:`OpenAICompatProvider`

    Pass a provider instance explicitly to
    :class:`~omop_llm.interface.client.LLMClient` to override this inference
    for custom or future backends.

    Parameters
    ----------
    api_base : str
        Base URL of the API endpoint.
    api_key : str, optional
        API key, used as a secondary Ollama signal when the URL alone is
        ambiguous (e.g. a plain ``localhost`` URL).  Default is ``'ollama'``.

    Returns
    -------
    EmbeddingProvider
        A provider instance appropriate for *api_base*.
    """
    is_local = "localhost" in api_base or "127.0.0.1" in api_base
    is_ollama = "ollama" in api_base or (is_local and api_key == "ollama")

    if is_ollama:
        return OllamaProvider()
    return OpenAICompatProvider()
