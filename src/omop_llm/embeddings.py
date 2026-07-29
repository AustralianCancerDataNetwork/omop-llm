"""Embedding role prefixing: ``EmbeddingRole`` and a best-effort sanity check.

Asymmetric embedding models (nomic-embed-text, the E5 family, BGE, and
others) are trained with distinct prefixes for the text being indexed
versus the text used to search it. Sending text without the correct
prefix produces a valid-looking embedding that just retrieves badly,
with no error to notice.
"""

from __future__ import annotations

import logging
from enum import StrEnum
from typing import Any

logger = logging.getLogger(__name__)


class EmbeddingRole(StrEnum):
    """Role of text being embedded, for models with asymmetric prefixes."""

    DOCUMENT = "document"
    QUERY = "query"


CONFIGURATION_KEY_BY_ROLE: dict[EmbeddingRole, str] = {
    EmbeddingRole.DOCUMENT: "document_prefix",
    EmbeddingRole.QUERY: "query_prefix",
}

# Prefix conventions used by common asymmetric embedding models. Not
# exhaustive, and not meant to be: used only to flag a configured prefix
# that doesn't match anything recognized, never to reject or "correct"
# one. New, legitimate conventions turn up regularly; extend this set
# as they're encountered rather than trying to gate on it.
KNOWN_EMBEDDING_PREFIXES: frozenset[str] = frozenset(
    {
        "search_document: ",
        "search_query: ",
        "passage: ",
        "query: ",
        "Represent this sentence for searching relevant passages: ",
    }
)


def apply_embedding_prefix(
    texts: list[str], role: EmbeddingRole, configuration: dict[str, Any]
) -> list[str]:
    """Prepend *role*'s configured prefix to each of *texts*, if one is set."""
    prefix = configuration.get(CONFIGURATION_KEY_BY_ROLE[role], "")
    if not prefix:
        return texts
    return [f"{prefix}{text}" for text in texts]


def warn_if_prefixes_look_wrong(*, model: str, configuration: dict[str, Any]) -> None:
    """Log a warning for a missing or unrecognized configured prefix.

    Called once, at :func:`~omop_llm.backend.build_model_backend` time, for
    any backend that declares embeddings support. Never raises: a prefix
    outside :data:`KNOWN_EMBEDDING_PREFIXES` is not necessarily wrong, this
    is a heads-up, not validation.
    """
    for role, key in CONFIGURATION_KEY_BY_ROLE.items():
        prefix = configuration.get(key)
        if not prefix:
            logger.warning(
                "%s: no %s configured for model %r. Fine for symmetric models; "
                "asymmetric models (e.g. nomic-embed-text, E5, BGE) need one "
                "to retrieve correctly.",
                role.value.capitalize(), key, model,
            )
        elif prefix not in KNOWN_EMBEDDING_PREFIXES:
            logger.warning(
                "%s prefix %r for model %r doesn't match a commonly recognized "
                "convention. Not necessarily wrong, just worth double-checking.",
                role.value.capitalize(), prefix, model,
            )
