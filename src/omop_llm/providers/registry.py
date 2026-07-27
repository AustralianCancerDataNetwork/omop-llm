"""The closed set of providers omop_llm exposes.

any-llm itself supports around fifty providers; omop_llm intentionally
supports six, matched to what this stack actually runs: local (``ollama``,
``llama-server`` via ``llamacpp``, ``vllm``) and cloud (``openai``,
``anthropic``, ``gemini``). See :mod:`omop_llm.providers.supported` for
the six classes themselves. A provider not defined there is structurally
unreachable through omop_llm's public API, regardless of what any-llm
itself supports.

``PROVIDER_REGISTRY`` is built by discovering
:class:`~omop_llm.providers.base.ProviderMixin`'s own subclasses,
not by a second, separately-maintained list of classes: the set of
supported providers is defined exactly once, in
:mod:`omop_llm.providers.supported`, and this module can't drift out of
sync with it because it has nothing of its own to drift. One caveat that
comes with discovery over a class registry: any other direct subclass of
``ProviderMixin`` loaded into the process (e.g. a test fixture)
would also appear here. Nothing in this package does that; if a test ever
needs a fake provider, it should not subclass the mixin directly.
"""

from __future__ import annotations

from typing import Final

from any_llm.any_llm import AnyLLM

from omop_llm.capabilities import ModelCapabilities
from omop_llm.errors import UnsupportedProviderError
from omop_llm.providers import supported as _supported  # noqa: F401  (required for PROVIDER_REGISTRY to be populated)
from omop_llm.providers.base import ProviderMixin

PROVIDER_REGISTRY: Final[dict[str, type[AnyLLM]]] = {
    cls.PROVIDER_NAME: cls
    for cls in ProviderMixin.__subclasses__()
    if issubclass(cls, AnyLLM)
}


def supported_providers() -> tuple[str, ...]:
    """List the provider keys omop_llm will construct a backend for.

    Returns
    -------
    tuple of str
        The registered provider keys, sorted alphabetically.
    """
    return tuple(sorted(PROVIDER_REGISTRY))


def provider_class_for(provider_key: str) -> type[AnyLLM]:
    """Look up a registered provider class.

    Parameters
    ----------
    provider_key : str
        A key expected to be in :data:`PROVIDER_REGISTRY`.

    Returns
    -------
    type of AnyLLM
        The provider class registered for ``provider_key``.

    Raises
    ------
    UnsupportedProviderError
        If ``provider_key`` is not registered.
    """
    try:
        return PROVIDER_REGISTRY[provider_key]
    except KeyError:
        raise UnsupportedProviderError(
            f"{provider_key!r} is not a supported provider. "
            f"Supported: {', '.join(supported_providers())}"
        ) from None


def capabilities_for(provider_key: str) -> ModelCapabilities:
    """Build the capability declaration for one registered provider.

    ``streaming``, ``embeddings``, and ``extended_thinking`` come straight
    from any-llm's own ``get_provider_metadata()``. ``tool_use`` and
    ``structured_output`` come from the class attributes each provider
    subclass declares itself, since any-llm tracks neither.

    Parameters
    ----------
    provider_key : str
        A key expected to be in :data:`PROVIDER_REGISTRY`.

    Returns
    -------
    ModelCapabilities
        The capability declaration for this provider.

    Raises
    ------
    UnsupportedProviderError
        If ``provider_key`` is not registered.
    """
    provider_class = provider_class_for(provider_key)
    meta = provider_class.get_provider_metadata()
    assert issubclass(provider_class, ProviderMixin)
    return ModelCapabilities(
        streaming=meta.streaming,
        embeddings=meta.embedding,
        extended_thinking=meta.reasoning,
        tool_use=provider_class.TOOL_USE,
        structured_output=provider_class.STRUCTURED_OUTPUT,
    )


def canonical_model_name(provider_key: str, name: str) -> str:
    """Canonicalize a model name for one registered provider.

    Useful for deciding what to persist as a model's stable identity (e.g.
    in a database) independently of building a full
    :class:`~omop_llm.backend.ModelBackend`. :func:`~omop_llm.backend.build_backend`
    also calls this internally, so a backend's ``model`` attribute is
    always canonical without callers needing to remember to do it
    themselves.

    Parameters
    ----------
    provider_key : str
        A key expected to be in :data:`PROVIDER_REGISTRY`.
    name : str
        Raw model name to canonicalize.

    Returns
    -------
    str
        The canonical model name for this provider.

    Raises
    ------
    UnsupportedProviderError
        If ``provider_key`` is not registered.
    ValueError
        If ``name`` cannot be made canonical for this provider (e.g. an
        Ollama name with no explicit tag).
    """
    provider_class = provider_class_for(provider_key)
    assert issubclass(provider_class, ProviderMixin)
    return provider_class.canonical_model_name(name)
