"""Optional fallback for structured extraction: ``instructor``'s validate-and-retry loop.

The primary structured-extraction path lives on
:meth:`omop_llm.backend.ModelBackend.extract`/:meth:`~omop_llm.backend.ModelBackend.async_extract`,
built directly on any-llm's own ``response_format=<PydanticModel>``
passthrough. This module is a separate, explicitly scoped alternative for
callers that specifically want resilience against a model returning
almost-valid JSON, kept out of ``backend.py`` so importing ``omop_llm``
never requires the optional ``instructor`` dependency.

It is *not* wired in as a silent alternative for every provider. This was
checked directly against ``instructor``'s own source
(``instructor.v2.auto_client._PROVIDER_BUILDERS``):

- ``ollama`` is not safe to route through it: instructor's own Ollama
  builder constructs a plain ``openai.AsyncOpenAI(base_url=".../v1")``
  client, the OpenAI-compat shim, not native ``/api/chat``, and picks
  TOOLS-vs-JSON mode from a hardcoded model-name-substring list (the exact
  "guess capability from the model name" anti-pattern this whole package
  exists to retire). Using it for ``ollama`` would silently regress the
  native-transport fidelity ``cava-nlp-shard`` depends on today.
- ``llamacpp``/``vllm`` have no dedicated builder in ``instructor`` at all
  (its provider list tops out at roughly 23 hosted vendors). They are
  reachable only by routing through instructor's ``openai`` builder with
  an explicit ``base_url`` override, which is what
  :func:`extract_with_retry`/:func:`async_extract_with_retry` do.
- ``anthropic``/``gemini`` are not offered here either: this module only
  vouches for providers whose any-llm integration is already
  OpenAI-compat-native, so there is no native-transport distinction to
  lose. Requesting anything outside ``{"openai", "llamacpp", "vllm"}``
  raises :class:`~omop_llm.errors.UnsupportedCapabilityError` rather than
  silently downgrading transport.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from omop_llm.errors import UnsupportedCapabilityError
from omop_llm.providers.supported import LlamacppProvider, OpenaiProvider, VllmProvider

_INSTRUCTOR_SAFE_PROVIDERS = frozenset({
    OpenaiProvider.PROVIDER_NAME, 
    LlamacppProvider.PROVIDER_NAME, 
    VllmProvider.PROVIDER_NAME
})


def _check_provider_and_base_url(provider: str, base_url: str | None) -> None:
    if provider not in _INSTRUCTOR_SAFE_PROVIDERS:
        raise UnsupportedCapabilityError(
            f"instructor-based extraction is not offered for provider {provider!r}: "
            f"only {sorted(_INSTRUCTOR_SAFE_PROVIDERS)} are confirmed to share any-llm's "
            "transport for this provider, see the omop_llm.structured module docstring"
        )
    if provider != "openai" and base_url is None:
        raise ValueError(
            f"base_url is required for provider={provider!r} "
            "(without it, instructor's 'openai' builder would silently target "
            "the real OpenAI API instead of your local/TRE server)"
        )


def _require_instructor() -> Any:
    try:
        import instructor
    except ImportError as exc:
        raise UnsupportedCapabilityError(
            "instructor-based extraction requires the 'instructor' optional extra: "
            "pip install 'omop-llm[instructor]'"
        ) from exc
    return instructor


def extract_with_retry[T: BaseModel](
    provider: str,
    model: str,
    messages: list[dict[str, Any]],
    response_model: type[T],
    *,
    base_url: str | None = None,
    api_key: str | None = None,
    max_retries: int = 2,
    **kwargs: Any,
) -> T:
    """Extract via ``instructor``'s validate-and-retry loop, synchronously.

    See :func:`async_extract_with_retry` for parameters.
    """
    _check_provider_and_base_url(provider, base_url)
    instructor = _require_instructor()

    client_kwargs: dict[str, Any] = {"async_client": False}
    if base_url is not None:
        client_kwargs["base_url"] = base_url
    if api_key is not None:
        client_kwargs["api_key"] = api_key

    client = instructor.from_provider(f"openai/{model}", **client_kwargs)
    return client.chat.completions.create(
        messages=messages,
        response_model=response_model,
        max_retries=max_retries,
        **kwargs,
    )


async def async_extract_with_retry[T: BaseModel](
    provider: str,
    model: str,
    messages: list[dict[str, Any]],
    response_model: type[T],
    *,
    base_url: str | None = None,
    api_key: str | None = None,
    max_retries: int = 2,
    **kwargs: Any,
) -> T:
    """Extract via ``instructor``'s validate-and-retry loop.

    Requires the ``instructor`` optional extra
    (``pip install 'omop-llm[instructor]'``).

    Parameters
    ----------
    provider : str
        One of ``{"openai", "llamacpp", "vllm"}`` (see module docstring).
        ``llamacpp``/``vllm`` are routed through ``instructor``'s
        ``openai`` builder with an explicit ``base_url``, which is
        therefore required for those two, to avoid silently falling back
        to instructor's real-OpenAI default endpoint.
    model : str
        The model name or identifier.
    messages : list of dict
        Chat history in OpenAI message format.
    response_model : type of BaseModel
        The Pydantic model to constrain and validate the response against.
    base_url : str, optional
        The provider's base URL. Required when ``provider`` is not
        ``"openai"``.
    api_key : str, optional
        The API key for this provider, if one is required.
    max_retries : int, optional
        Number of validate-and-retry attempts. Default is 2.
    **kwargs : Any
        Additional arguments forwarded to instructor's
        ``chat.completions.create``.

    Returns
    -------
    BaseModel
        A validated instance of ``response_model``.

    Raises
    ------
    UnsupportedCapabilityError
        If ``provider`` is not in ``{"openai", "llamacpp", "vllm"}``, or if
        the ``instructor`` optional extra is not installed.
    ValueError
        If ``provider`` is not ``"openai"`` and ``base_url`` is not given.
    """
    _check_provider_and_base_url(provider, base_url)
    instructor = _require_instructor()

    client_kwargs: dict[str, Any] = {"async_client": True}
    if base_url is not None:
        client_kwargs["base_url"] = base_url
    if api_key is not None:
        client_kwargs["api_key"] = api_key

    client = instructor.from_provider(f"openai/{model}", **client_kwargs)
    return await client.chat.completions.create(
        messages=messages,
        response_model=response_model,
        max_retries=max_retries,
        **kwargs,
    )
