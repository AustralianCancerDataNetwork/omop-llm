"""Shared base for omop_llm's own provider subclasses."""

from __future__ import annotations

from abc import ABC, abstractmethod


class ProviderMixin(ABC):
    """Marks a class as one of omop_llm's own provider subclasses.

    Declares the two capabilities any-llm does not track itself
    (``TOOL_USE``, ``STRUCTURED_OUTPUT``) as class attributes, alongside
    any-llm's own ``SUPPORTS_*`` flags on the sibling base class. Also
    declares the two provider-specific hooks a resolved backend needs:
    :meth:`canonical_model_name` and :meth:`embedding_dimension_hint`.

    Every provider omop_llm supports gets a real subclass built on this
    mixin. Any-llm's own base class, ``AnyLLM``, is already an ``abc.ABC``
    with real abstract methods, so this mixin composes with it safely.
    ``canonical_model_name`` is a required override, not a default
    passthrough, so adding a new provider forces a deliberate decision
    about its naming rules rather than silently inheriting "no
    transformation needed."

    Attributes
    ----------
    TOOL_USE : bool
        Whether this provider supports tool/function calling.
    STRUCTURED_OUTPUT : bool
        Whether this provider supports structured (schema-constrained)
        output.
    """

    TOOL_USE: bool
    STRUCTURED_OUTPUT: bool

    @classmethod
    @abstractmethod
    def canonical_model_name(cls, name: str) -> str:
        """Return the canonical form of a model name for this provider.

        The canonical form is the identifier used as a stable key
        wherever a consumer persists model identity (e.g. ``omop-emb``'s
        embedding registry), and the ``model`` value
        :func:`~omop_llm.backend.build_backend` resolves to. Implementations
        must be idempotent: calling this on an already-canonical name
        returns the same string unchanged.

        Parameters
        ----------
        name : str
            Raw model name as supplied by the caller, e.g. ``"llama3"`` or
            ``"text-embedding-3-small"``.

        Returns
        -------
        str
            The canonical model name for this provider.

        Raises
        ------
        ValueError
            If the name cannot be made canonical (e.g. an Ollama name with
            no explicit tag).
        """
        ...

    async def async_embedding_dimension_hint(self, model: str, *, api_base: str | None) -> int | None:
        """Look up this model's embedding dimension via a provider-specific fast path.

        Default: no fast path available. Override where a provider
        exposes model metadata directly (e.g. Ollama's ``POST /api/show``).
        Used as the middle tier of :meth:`omop_llm.backend.ModelBackend.async_dimensions`,
        between a configured override and a live embedding probe.

        Parameters
        ----------
        model : str
            The canonical model name.
        api_base : str, optional
            The resolved base URL this backend was constructed with. May
            be ``None`` if it was not explicitly configured; providers
            that need it to build a fast-path request should return
            ``None`` in that case rather than guessing a default.

        Returns
        -------
        int or None
            The embedding dimension, or ``None`` if this provider has no
            fast path for it.
        """
        return None

    def embedding_dimension_hint(self, model: str, *, api_base: str | None) -> int | None:
        """Synchronous counterpart to :meth:`async_embedding_dimension_hint`.

        Parameters
        ----------
        model : str
            The canonical model name.
        api_base : str, optional
            The resolved base URL this backend was constructed with. See
            :meth:`async_embedding_dimension_hint`.

        Returns
        -------
        int or None
            The embedding dimension, or ``None`` if this provider has no
            fast path for it.
        """
        return None
