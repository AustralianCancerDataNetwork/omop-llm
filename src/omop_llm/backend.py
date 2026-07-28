"""``ModelBackend``: the one calling contract every consumer uses.

A thin wrapper around a single, already-constructed any-llm provider
instance (an entry of :data:`omop_llm.providers.registry.PROVIDER_REGISTRY`).
Chat completion, embeddings, and structured extraction are all methods on
one object, gated by :class:`~omop_llm.capabilities.ModelCapabilities`,
rather than split across separate classes per modality.

Every method has a synchronous form and an ``async_``-prefixed
asynchronous form (``complete``/``async_complete``,
``embed_texts``/``async_embed_texts``, and so on). This was a deliberate
choice, not an oversight. 

Consumers only ever see this class, never a raw any-llm provider instance.
If any-llm needed replacing, only this module's method bodies, and the
``providers/`` subclasses, would need to change.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

from any_llm.any_llm import AnyLLM
from any_llm.types.completion import ChatCompletion, ReasoningEffort
from oa_configurator import ResolvedModel
from pydantic import BaseModel, ValidationError

from omop_llm.capabilities import ModelCapabilities
from omop_llm.errors import NoParsedOutputError, UnsupportedCapabilityError
from omop_llm.providers.base import ProviderMixin
from omop_llm.providers.registry import (
    canonical_model_name,
    capabilities_for,
    provider_class_for,
)


def _chunked[T](items: list[T], size: int) -> Iterator[list[T]]:
    """Yield successive sub-lists of ``items``, each at most ``size`` long.

    Parameters
    ----------
    items : list
        The items to chunk.
    size : int
        Maximum length of each yielded chunk. Must be positive.

    Raises
    ------
    ValueError
        If ``size`` is not a positive integer.
    """
    if size <= 0:
        raise ValueError(f"batch_size must be a positive integer, got {size!r}")
    for start in range(0, len(items), size):
        yield items[start : start + size]


@dataclass
class ModelBackend:
    """One resolved, ready-to-call model.

    Built by :func:`build_backend`. Wraps a single constructed any-llm
    provider instance and binds ``model``/``configuration`` to it, so
    callers do not repeat them on every call.

    Parameters
    ----------
    _client : AnyLLM
        The constructed any-llm provider instance backing this backend.
    model : str
        The canonical model name or identifier passed to the underlying
        provider.
    capabilities : ModelCapabilities
        What this resolved backend can actually do.
    configuration : dict, optional
        Default keyword arguments merged into every call, overridden by
        any argument the caller passes explicitly.
    _api_base : str, optional
        The base URL this backend was constructed with, if any. Threaded
        through to provider-specific fast paths such as
        :meth:`~omop_llm.providers.supported.OllamaProvider.embedding_dimension_hint`.
    """

    _client: AnyLLM
    model: str
    capabilities: ModelCapabilities
    configuration: dict[str, Any] = field(default_factory=dict)
    _api_base: str | None = None

    @property
    def provider(self) -> str:
        """The provider key this backend was resolved to, e.g. ``"ollama"``.

        Read directly off ``_client``'s own any-llm ``PROVIDER_NAME`` class
        attribute rather than stored separately at construction time, so
        there is exactly one place this string is ever defined (see
        :data:`omop_llm.providers.registry.PROVIDER_REGISTRY`, whose keys
        are derived from the same attribute).
        """
        return self._client.PROVIDER_NAME

    def _build_call_kwargs(
        self,
        *,
        tools: list[dict[str, Any]] | None,
        response_format: dict[str, Any] | type | None,
        max_tokens: int | None,
        temperature: float | None,
        reasoning_effort: ReasoningEffort | None,
        extra: dict[str, Any],
    ) -> dict[str, Any]:
        if extra.get("stream"):
            raise NotImplementedError(
                "ModelBackend does not support stream=True yet; "
                "complete()/async_complete() are typed to always return a "
                "ChatCompletion, not a chunk iterator"
            )
        call_kwargs: dict[str, Any] = {**self.configuration, **extra}
        if tools is not None:
            call_kwargs["tools"] = tools
        if response_format is not None:
            call_kwargs["response_format"] = response_format
        if max_tokens is not None:
            call_kwargs["max_tokens"] = max_tokens
        if temperature is not None:
            call_kwargs["temperature"] = temperature
        if reasoning_effort is not None:
            call_kwargs["reasoning_effort"] = reasoning_effort
        return call_kwargs

    def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        response_format: dict[str, Any] | type | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        reasoning_effort: ReasoningEffort | None = None,
        **kwargs: Any,
    ) -> ChatCompletion:
        """Run one chat completion synchronously. See :meth:`async_complete` for parameters."""
        call_kwargs = self._build_call_kwargs(
            tools=tools, response_format=response_format, max_tokens=max_tokens,
            temperature=temperature, reasoning_effort=reasoning_effort, extra=kwargs,
        )
        # always non-streaming and typed to return a ChatCompletion
        # but not captured by any single any-llm overload
        return self._client.completion(  # ty: ignore[no-matching-overload]
            model=self.model, messages=messages, **call_kwargs
        )

    async def async_complete(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        response_format: dict[str, Any] | type | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        reasoning_effort: ReasoningEffort | None = None,
        **kwargs: Any,
    ) -> ChatCompletion:
        """Run one chat completion.

        Parameters
        ----------
        messages : list of dict
            Chat history in OpenAI message format.
        tools : list of dict, optional
            Raw OpenAI-style tool schema list. any-llm normalizes tool-call
            parsing per provider, so callers doing multi-turn agentic tool
            use pass the same schema regardless of which provider is
            resolved. Requires ``self.capabilities.tool_use``.
        response_format : dict or type, optional
            A raw JSON-schema dict, or a Pydantic model class. any-llm
            translates a Pydantic class into each provider's own native
            structured-output mechanism. See :meth:`extract`/:meth:`async_extract`
            for a convenience method that validates and unwraps the
            result.
        max_tokens : int, optional
            Maximum number of tokens to generate.
        temperature : float, optional
            Sampling temperature.
        reasoning_effort : ReasoningEffort, optional
            Requested extended-thinking effort, any-llm's own normalized
            parameter across providers. Only meaningful when
            ``self.capabilities.extended_thinking`` is ``True``; a provider
            without reasoning support ignores it.
        **kwargs : Any
            Additional provider-specific arguments, passed through
            unchanged. ``stream`` is rejected: this method always returns
            a ``ChatCompletion``, never a chunk iterator, and streaming is
            not designed or supported here yet.

        Returns
        -------
        ChatCompletion
            The completion response.
        """
        call_kwargs = self._build_call_kwargs(
            tools=tools, response_format=response_format, max_tokens=max_tokens,
            temperature=temperature, reasoning_effort=reasoning_effort, extra=kwargs,
        )
        # See the matching comment in complete(): our response_format
        # union doesn't match any single any-llm acompletion() overload.
        return await self._client.acompletion(  # ty: ignore[no-matching-overload]
            model=self.model, messages=messages, **call_kwargs
        )

    def embed_texts(self, texts: list[str], *, batch_size: int | None = None) -> list[list[float]]:
        """Embed a batch of texts.

        Parameters
        ----------
        texts : list of str
            Texts to embed.
        batch_size : int, optional
            If given, ``texts`` is chunked into sub-batches of at most this
            size, each sent as its own call, rather than one call with the
            entire list. Useful for bulk callers embedding more texts than
            a single provider request should carry. Default is ``None``
            (one call for the whole list).

        Returns
        -------
        list of list of float
            One embedding vector per input text, in the same order.

        Raises
        ------
        UnsupportedCapabilityError
            If ``self.capabilities.embeddings`` is ``False``.
        ValueError
            If ``batch_size`` is not a positive integer.
        """
        self._require_embeddings()
        if batch_size is None:
            response = self._client._embedding(model=self.model, inputs=texts, **self.configuration)
            return [item.embedding for item in response.data]
        vectors: list[list[float]] = []
        for chunk in _chunked(texts, batch_size):
            response = self._client._embedding(model=self.model, inputs=chunk, **self.configuration)
            vectors.extend(item.embedding for item in response.data)
        return vectors

    async def async_embed_texts(
        self, texts: list[str], *, batch_size: int | None = None
    ) -> list[list[float]]:
        """Embed a batch of texts asynchronously.

        Parameters
        ----------
        texts : list of str
            Texts to embed.
        batch_size : int, optional
            If given, ``texts`` is chunked into sub-batches of at most this
            size, each sent as its own call, rather than one call with the
            entire list. Useful for bulk callers embedding more texts than
            a single provider request should carry. Default is ``None``
            (one call for the whole list).

        Returns
        -------
        list of list of float
            One embedding vector per input text, in the same order.

        Raises
        ------
        UnsupportedCapabilityError
            If ``self.capabilities.embeddings`` is ``False``.
        ValueError
            If ``batch_size`` is not a positive integer.
        """
        self._require_embeddings()
        if batch_size is None:
            response = await self._client.aembedding(model=self.model, inputs=texts, **self.configuration)
            return [item.embedding for item in response.data]
        vectors: list[list[float]] = []
        for chunk in _chunked(texts, batch_size):
            response = await self._client.aembedding(model=self.model, inputs=chunk, **self.configuration)
            vectors.extend(item.embedding for item in response.data)
        return vectors

    def _require_embeddings(self) -> None:
        if not self.capabilities.embeddings:
            raise UnsupportedCapabilityError(
                f"backend for model {self.model!r} does not support embeddings"
            )

    def dimensions(self) -> int:
        """Discover this model's embedding dimensionality synchronously.
        Three tiers: 
            1. a configured override (``configuration["embedding_dim"]``),
            2. a provider-specific fast path (e.g. Ollama's ``POST /api/show``), and
            3. a live probe (embed one short string and measure the vector).

        Returns
        -------
        int
            The embedding vector length.
        """
        configured = self.configuration.get("embedding_dim")
        if configured is not None:
            return int(configured)
        assert isinstance(self._client, ProviderMixin)
        hint = self._client.embedding_dimension_hint(self.model, api_base=self._api_base)
        if hint is not None:
            return hint
        [vector] = self.embed_texts(["dimension probe"])
        return len(vector)

    async def async_dimensions(self) -> int:
        """Discover this model's embedding dimensionality.
        Three tiers: 
            1. a configured override (``configuration["embedding_dim"]``),
            2. a provider-specific fast path (e.g. Ollama's ``POST /api/show``), and
            3. a live probe (embed one short string and measure the vector).

        Returns
        -------
        int
            The embedding vector length.
        """
        configured = self.configuration.get("embedding_dim")
        if configured is not None:
            return int(configured)
        assert isinstance(self._client, ProviderMixin)
        hint = await self._client.async_embedding_dimension_hint(self.model, api_base=self._api_base)
        if hint is not None:
            return hint
        [vector] = await self.async_embed_texts(["dimension probe"])
        return len(vector)

    def extract[T: BaseModel](
        self,
        messages: list[dict[str, Any]],
        response_model: type[T],
        *,
        max_retries: int = 0,
        **kwargs: Any,
    ) -> T:
        """Extract one validated ``response_model`` instance from a chat call synchronously.

        A thin convenience method built on :meth:`complete` with
        ``response_format=response_model``. Checks that a parsed instance
        actually came back, and unwraps it.

        Notes
        -----
        ``max_retries`` is native (not `instructor`-based), so it works for
        all providers, unlike :func:`omop_llm.structured.extract_with_retry`.

        Parameters
        ----------
        messages : list of dict
            Chat history in OpenAI message format.
        response_model : type of BaseModel
            The Pydantic model to constrain and validate the response
            against.
        max_retries : int, optional
            Number of additional attempts after a validation failure.
            Default is 0, i.e. fail immediately.
        **kwargs : Any
            Additional arguments forwarded to :meth:`complete`.

        Returns
        -------
        BaseModel
            A validated instance of ``response_model``.

        Raises
        ------
        UnsupportedCapabilityError
            If ``self.capabilities.structured_output`` is ``False``.
        NoParsedOutputError
            If the provider returned no parsed instance (refusal or empty
            content), after exhausting ``max_retries``.
        any_llm.exceptions.LengthFinishReasonError
            If the response was truncated before completing.
        any_llm.exceptions.ContentFilterFinishReasonError
            If a content filter blocked the response.
        pydantic.ValidationError
            If the model's output does not match ``response_model``'s
            schema, after exhausting ``max_retries``.
        """
        self._require_structured_output()
        conversation = list(messages)
        for attempt in range(max_retries + 1):
            try:
                completion = self.complete(conversation, response_format=response_model, **kwargs)
                return self._unwrap_parsed(completion, response_model)
            except (NoParsedOutputError, ValidationError) as exc:
                if attempt >= max_retries:
                    raise
                conversation = [*conversation, self._retry_extract_message(exc, response_model)]
        raise AssertionError("unreachable")

    async def async_extract[T: BaseModel](
        self,
        messages: list[dict[str, Any]],
        response_model: type[T],
        *,
        max_retries: int = 0,
        **kwargs: Any,
    ) -> T:
        """Extract one validated ``response_model`` instance from a chat call.

        A thin convenience method built on :meth:`async_complete` with
        ``response_format=response_model``. Checks that a parsed instance
        actually came back, and unwraps it.

        Notes
        -----
        ``max_retries`` is native (not `instructor`-based), so it works for
        all providers, unlike :func:`omop_llm.structured.extract_with_retry`.

        Parameters
        ----------
        messages : list of dict
            Chat history in OpenAI message format.
        response_model : type of BaseModel
            The Pydantic model to constrain and validate the response
            against.
        max_retries : int, optional
            Number of additional attempts after a validation failure.
            Default is 0, i.e. fail immediately.
        **kwargs : Any
            Additional arguments forwarded to :meth:`async_complete`.

        Returns
        -------
        BaseModel
            A validated instance of ``response_model``.

        Raises
        ------
        UnsupportedCapabilityError
            If ``self.capabilities.structured_output`` is ``False``.
        NoParsedOutputError
            If the provider returned no parsed instance (refusal or empty
            content), after exhausting ``max_retries``.
        any_llm.exceptions.LengthFinishReasonError
            If the response was truncated before completing.
        any_llm.exceptions.ContentFilterFinishReasonError
            If a content filter blocked the response.
        pydantic.ValidationError
            If the model's output does not match ``response_model``'s
            schema, after exhausting ``max_retries``.
        """
        self._require_structured_output()
        conversation = list(messages)
        for attempt in range(max_retries + 1):
            try:
                completion = await self.async_complete(conversation, response_format=response_model, **kwargs)
                return self._unwrap_parsed(completion, response_model)
            except (NoParsedOutputError, ValidationError) as exc:
                if attempt >= max_retries:
                    raise
                conversation = [*conversation, self._retry_extract_message(exc, response_model)]
        raise AssertionError("unreachable")

    def _require_structured_output(self) -> None:
        if not self.capabilities.structured_output:
            raise UnsupportedCapabilityError(
                f"backend for model {self.model!r} does not declare structured_output support"
            )

    @staticmethod
    def _retry_extract_message[T: BaseModel](exc: Exception, response_model: type[T]) -> dict[str, Any]:
        """Build a follow-up user message asking the model to correct a failed extraction."""
        return {
            "role": "user",
            "content": (
                f"Your previous response was invalid: {exc}. "
                f"Respond again with valid JSON matching {response_model.__name__}'s schema."
            ),
        }

    @staticmethod
    def _unwrap_parsed[T: BaseModel](completion: ChatCompletion, response_model: type[T]) -> T:
        """Unwrap a completion's parsed instance, or raise if there is none to unwrap.

        Parameters
        ----------
        completion : ChatCompletion
            The completion returned from a call made with
            ``response_format=response_model``.
        response_model : type of BaseModel
            The Pydantic model the caller expected back.

        Returns
        -------
        BaseModel
            The validated ``response_model`` instance any-llm attached to
            the completion.

        Raises
        ------
        NoParsedOutputError
            If no parsed instance is present.
        """
        message = completion.choices[0].message
        parsed = getattr(message, "parsed", None)
        if parsed is None:
            raise NoParsedOutputError(
                f"provider returned no parsed {response_model.__name__} instance "
                "(no validation or length/content-filter error was raised, so the "
                "model likely refused to answer or returned empty content)"
            )
        return parsed  # type: ignore[no-any-return]

    def is_available(self, **kwargs: Any) -> bool:
        """Check whether this backend can actually be reached, synchronously.

        Probes ``list_models`` against the resolved provider. Swallows any
        error and reports ``False`` rather than raising, since the point of
        a health check is to answer "can I use this," not to propagate the
        specific failure.

        Parameters
        ----------
        **kwargs : Any
            Forwarded to the underlying ``list_models`` call, e.g.
            ``timeout=2.0``.

        Returns
        -------
        bool
            Whether listing models against this backend succeeded.
        """
        try:
            self._client.list_models(**kwargs)
        except Exception:  # noqa: BLE001 (any exception means "unavailable")
            return False
        return True

    async def async_is_available(self, **kwargs: Any) -> bool:
        """Check whether this backend can actually be reached, asynchronously.

        Probes ``list_models`` against the resolved provider. Swallows any
        error and reports ``False`` rather than raising, since the point of
        a health check is to answer "can I use this," not to propagate the
        specific failure.

        Parameters
        ----------
        **kwargs : Any
            Forwarded to the underlying ``alist_models`` call, e.g.
            ``timeout=2.0``.

        Returns
        -------
        bool
            Whether listing models against this backend succeeded.
        """
        try:
            await self._client.alist_models(**kwargs)
        except Exception:  # noqa: BLE001 (any exception means "unavailable")
            return False
        return True


def build_backend(
    provider: str,
    model: str,
    *,
    base_url: str | None = None,
    api_key: str | None = None,
    configuration: dict[str, Any] | None = None,
) -> ModelBackend:
    """Resolve a provider and model into a ready-to-call backend.

    Plain keyword arguments in, a :class:`ModelBackend` out, mirroring the
    shape ``oa-configurator``'s own database resolution already uses
    (``Resolver(stack).resolve_resource(name).create_engine(**kwargs)``
    returns a plain ``sqlalchemy.Engine``, no intermediate config object).
    See :func:`build_backend_from_resolved` for the ``oa-configurator``
    integration built on top of this function.

    Canonicalizes ``model`` for the resolved provider (see
    :func:`omop_llm.providers.registry.canonical_model_name`), so a
    :class:`ModelBackend`'s ``model`` attribute is always canonical.

    Parameters
    ----------
    provider : str
        A key in :data:`omop_llm.providers.registry.PROVIDER_REGISTRY`.
    model : str
        Raw model name or identifier; canonicalized before use.
    base_url : str, optional
        The base URL for this specific deployment of the provider.
    api_key : str, optional
        The API key for this specific deployment, if one is required.
    configuration : dict, optional
        Default keyword arguments merged into every call this backend
        makes (e.g. ``max_tokens``, ``temperature``, ``embedding_dim``).

    Returns
    -------
    ModelBackend
        A backend ready to call, for example, :meth:`ModelBackend.complete`
        or :meth:`ModelBackend.async_complete`.

    Raises
    ------
    ValueError
        If ``model`` cannot be made canonical for the resolved provider
        (e.g. an Ollama name with no explicit tag).
    """
    provider_class = provider_class_for(provider)
    capabilities = capabilities_for(provider)
    canonical_model = canonical_model_name(provider, model)
    client = provider_class(api_key=api_key, api_base=base_url)
    return ModelBackend(
        _client=client,
        model=canonical_model,
        capabilities=capabilities,
        configuration=dict(configuration) if configuration else {},
        _api_base=base_url,
    )


def build_backend_from_resolved(resolved: ResolvedModel) -> ModelBackend:
    """Build a backend from an ``oa-configurator`` ``ResolvedModel``.

    The ``oa-configurator`` integration point: ``oa-configurator`` itself
    knows nothing about ``omop-llm`` (its ``ResolvedModel`` is plain data,
    the same way ``ResolvedResource`` is), so this glue lives here instead,
    mirroring ``omop_alchemy.config.create_cdm_engine(resolved: ResolvedResource) -> sa.Engine``:
    a consumer of ``oa-configurator`` takes its plain resolved output and
    does its own construction from it.

    A typical caller (e.g. a package's own config module) does::

        from oa_configurator import Resolver, load_stack_config
        from omop_llm import build_backend_from_resolved

        stack = load_stack_config()
        resolved = Resolver(stack).resolve_model(config.embedding_model)
        backend = build_backend_from_resolved(resolved)

    Parameters
    ----------
    resolved : ResolvedModel
        A model resolved via ``oa_configurator.Resolver.resolve_model()``.

    Returns
    -------
    ModelBackend
        A backend ready to call, for example, :meth:`ModelBackend.complete`
        or :meth:`ModelBackend.async_complete`.

    Raises
    ------
    ValueError
        If ``resolved.model`` cannot be made canonical for the resolved
        provider (e.g. an Ollama name with no explicit tag).
    """
    return build_backend(
        provider=resolved.provider.provider,
        model=resolved.model,
        base_url=resolved.provider.base_url,
        api_key=resolved.provider.api_key,
        configuration=resolved.configuration,
    )
