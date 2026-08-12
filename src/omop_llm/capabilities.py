"""What something -- a provider, a model, or a resolved backend -- can do."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Capabilities:
    """A capability declaration: what something can do.
    Used for providers, models, and resolved backends (the combination of the two).

    Opt-in: every field defaults to ``False``. Neither any-llm nor
    omop_llm can introspect a model's real capabilities, so nothing is
    assumed.

    Parameters
    ----------
    streaming : bool
        Whether streaming completions are supported.
    embeddings : bool
        Whether the embeddings endpoint is supported.
    extended_thinking : bool
        Whether reasoning/extended-thinking output is supported.
    tool_use : bool
        Whether tool/function calling is supported.
    structured_output : bool
        Whether structured (schema-constrained) output is supported.
    """

    streaming: bool = False
    embeddings: bool = False
    extended_thinking: bool = False
    tool_use: bool = False
    structured_output: bool = False

    def __and__(self, other: Capabilities) -> Capabilities:
        """Element-wise AND: a capability is only available if both sides have it."""
        return Capabilities(
            **{
                f.name: getattr(self, f.name) and getattr(other, f.name)
                for f in dataclasses.fields(self)
            }
        )
