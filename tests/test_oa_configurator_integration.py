"""``build_model_backend_from_resolved``: the oa-configurator integration point.

Constructs ``oa_configurator.ResolvedModel``/``ResolvedProvider`` directly
(no TOML file, no stack config needed) to test the field mapping in
isolation. Provider construction is offline throughout, no network access.
"""

from __future__ import annotations

from oa_configurator.resolver import ResolvedModel, ResolvedProvider

from omop_llm.backend import build_model_backend_from_resolved


def test_maps_resolved_fields_onto_build_backend() -> None:
    resolved = ResolvedModel(
        name="local-chat",
        provider=ResolvedProvider(
            name="local-llamacpp",
            provider="llamacpp",
            base_url="http://localhost:8080/v1",
            api_key=None,
        ),
        model="local-chat",
        embedding_dim=None,
        document_prefix=None,
        query_prefix=None,
        embeddings=True,
        tool_use=True,
        structured_output=True,
        extended_thinking=True,
        configuration={"max_tokens": 8000, "temperature": 0.0},
    )
    backend = build_model_backend_from_resolved(resolved)
    assert backend.model == "local-chat"
    assert backend.configuration == {"max_tokens": 8000, "temperature": 0.0}
    assert backend.capabilities.tool_use is True


def test_canonicalizes_the_model_name() -> None:
    resolved = ResolvedModel(
        name="m",
        provider=ResolvedProvider(name="p", provider="ollama", base_url="http://localhost:11434", api_key=None),
        model="llama3:8b",
        embedding_dim=None,
        document_prefix=None,
        query_prefix=None,
        embeddings=True,
        tool_use=True,
        structured_output=True,
        extended_thinking=True,
        configuration={},
    )
    backend = build_model_backend_from_resolved(resolved)
    assert backend.model == "llama3:8b"


def test_embedding_dim_and_prefixes_land_on_dedicated_fields_not_configuration() -> None:
    """The old behaviour folded these into `configuration`, which is exactly
    the bug that let them leak into the real provider call. They're now
    dedicated ModelBackend fields, and configuration stays untouched."""
    resolved = ResolvedModel(
        name="nomic-embed",
        provider=ResolvedProvider(name="p", provider="ollama", base_url="http://localhost:11434", api_key=None),
        model="nomic-embed-text:v1.5",
        embedding_dim=768,
        document_prefix="search_document: ",
        query_prefix="search_query: ",
        embeddings=True,
        tool_use=True,
        structured_output=True,
        extended_thinking=True,
        configuration={"max_tokens": 8000},
    )
    backend = build_model_backend_from_resolved(resolved)
    assert backend.embedding_dim == 768
    assert backend.document_prefix == "search_document: "
    assert backend.query_prefix == "search_query: "
    assert backend.configuration == {"max_tokens": 8000}


def test_configuration_dict_is_passed_through_untouched() -> None:
    """A same-named key already in resolved.configuration is just data now
    -- it's a coincidence, not a collision, since nothing merges into or
    reads out of `configuration` for these anymore."""
    resolved = ResolvedModel(
        name="nomic-embed",
        provider=ResolvedProvider(name="p", provider="ollama", base_url="http://localhost:11434", api_key=None),
        model="nomic-embed-text:v1.5",
        embedding_dim=768,
        document_prefix="search_document: ",
        query_prefix=None,
        embeddings=True,
        tool_use=True,
        structured_output=True,
        extended_thinking=True,
        configuration={"document_prefix": "stale: ", "query_prefix": "query: "},
    )
    backend = build_model_backend_from_resolved(resolved)
    assert backend.document_prefix == "search_document: "
    assert backend.configuration == {"document_prefix": "stale: ", "query_prefix": "query: "}


def test_capability_fields_narrow_the_provider_ceiling() -> None:
    """ollama's own provider metadata says embeddings=True; a model-level
    declaration of embeddings=False must still narrow the effective result,
    proving this is a real AND and not just reading the provider ceiling."""
    resolved = ResolvedModel(
        name="local-chat",
        provider=ResolvedProvider(name="p", provider="ollama", base_url="http://localhost:11434", api_key=None),
        model="local-chat:8b",
        embedding_dim=None,
        document_prefix=None,
        query_prefix=None,
        embeddings=False,
        tool_use=True,
        structured_output=True,
        extended_thinking=True,
        configuration={},
    )
    backend = build_model_backend_from_resolved(resolved)
    assert backend.capabilities.embeddings is False
