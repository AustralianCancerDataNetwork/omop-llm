"""What a resolved backend can actually do."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ModelCapabilities:
    """Capability declaration for one provider.

    ``streaming``, ``embeddings``, and ``extended_thinking`` are read
    directly from any-llm's own ``ProviderMetadata``.

    ``tool_use`` and ``structured_output`` have no equivalent in any-llm. 
    These two are declared by omop_llm itself in ``providers.registry`` and must not
    be inferred from any-llm's own introspection.

    Parameters
    ----------
    streaming : bool
        Whether the provider supports streaming completions.
    embeddings : bool
        Whether the provider supports the embeddings endpoint.
    extended_thinking : bool
        Whether the provider supports reasoning/extended-thinking output.
    tool_use : bool
        Whether the provider supports tool/function calling.
    structured_output : bool
        Whether the provider supports structured (schema-constrained)
        output.
    """

    streaming: bool
    embeddings: bool
    extended_thinking: bool
    tool_use: bool
    structured_output: bool
