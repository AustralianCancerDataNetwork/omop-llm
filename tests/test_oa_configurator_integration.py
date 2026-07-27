"""``build_backend_from_resolved``: the oa-configurator integration point.

Constructs ``oa_configurator.ResolvedModel``/``ResolvedProvider`` directly
(no TOML file, no stack config needed) to test the field mapping in
isolation. Provider construction is offline throughout, no network access.
"""

from __future__ import annotations

from oa_configurator.resolver import ResolvedModel, ResolvedProvider

from omop_llm.backend import build_backend_from_resolved


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
        configuration={"max_tokens": 8000, "temperature": 0.0},
    )
    backend = build_backend_from_resolved(resolved)
    assert backend.model == "local-chat"
    assert backend.configuration == {"max_tokens": 8000, "temperature": 0.0}
    assert backend.capabilities.tool_use is True


def test_canonicalizes_the_model_name() -> None:
    resolved = ResolvedModel(
        name="m",
        provider=ResolvedProvider(name="p", provider="ollama", base_url="http://localhost:11434", api_key=None),
        model="llama3:8b",
        configuration={},
    )
    backend = build_backend_from_resolved(resolved)
    assert backend.model == "llama3:8b"
