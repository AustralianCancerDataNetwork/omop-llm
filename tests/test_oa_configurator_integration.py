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
        configuration={},
    )
    backend = build_model_backend_from_resolved(resolved)
    assert backend.model == "llama3:8b"


def test_folds_embedding_dim_and_prefixes_into_configuration() -> None:
    resolved = ResolvedModel(
        name="nomic-embed",
        provider=ResolvedProvider(name="p", provider="ollama", base_url="http://localhost:11434", api_key=None),
        model="nomic-embed-text:v1.5",
        embedding_dim=768,
        document_prefix="search_document: ",
        query_prefix="search_query: ",
        configuration={"max_tokens": 8000},
    )
    backend = build_model_backend_from_resolved(resolved)
    assert backend.configuration == {
        "max_tokens": 8000,
        "embedding_dim": 768,
        "document_prefix": "search_document: ",
        "query_prefix": "search_query: ",
    }


def test_dedicated_fields_take_precedence_over_configuration_dict() -> None:
    resolved = ResolvedModel(
        name="nomic-embed",
        provider=ResolvedProvider(name="p", provider="ollama", base_url="http://localhost:11434", api_key=None),
        model="nomic-embed-text:v1.5",
        embedding_dim=768,
        document_prefix="search_document: ",
        query_prefix=None,
        configuration={"document_prefix": "stale: ", "query_prefix": "query: "},
    )
    backend = build_model_backend_from_resolved(resolved)
    assert backend.configuration["document_prefix"] == "search_document: "
    assert backend.configuration["query_prefix"] == "query: "
