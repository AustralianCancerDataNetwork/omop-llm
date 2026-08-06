"""EmbeddingRole prefixing and the known-prefix sanity check."""

from __future__ import annotations

import pytest

from omop_llm.embeddings import (
    KNOWN_EMBEDDING_PREFIXES,
    EmbeddingRole,
    apply_embedding_prefix,
    warn_if_prefixes_look_wrong,
)


class TestApplyEmbeddingPrefix:
    def test_document_prefix_applied(self) -> None:
        result = apply_embedding_prefix(
            ["diabetes"], EmbeddingRole.DOCUMENT, {"document_prefix": "passage: "}
        )
        assert result == ["passage: diabetes"]

    def test_query_prefix_applied(self) -> None:
        result = apply_embedding_prefix(
            ["hypertension"], EmbeddingRole.QUERY, {"query_prefix": "query: "}
        )
        assert result == ["query: hypertension"]

    def test_no_configured_prefix_is_a_noop(self) -> None:
        result = apply_embedding_prefix(["diabetes"], EmbeddingRole.DOCUMENT, {})
        assert result == ["diabetes"]

    def test_wrong_role_key_is_ignored(self) -> None:
        result = apply_embedding_prefix(
            ["diabetes"], EmbeddingRole.DOCUMENT, {"query_prefix": "query: "}
        )
        assert result == ["diabetes"]

    def test_applies_to_every_text(self) -> None:
        result = apply_embedding_prefix(
            ["a", "b", "c"], EmbeddingRole.DOCUMENT, {"document_prefix": "p: "}
        )
        assert result == ["p: a", "p: b", "p: c"]


class TestWarnIfPrefixesLookWrong:
    def test_warns_when_both_missing(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level("WARNING", logger="omop_llm.embeddings"):
            warn_if_prefixes_look_wrong(model="nomic-embed-text:v1.5", configuration={})
        assert "document_prefix" in caplog.text
        assert "query_prefix" in caplog.text

    def test_no_warning_when_both_known(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level("WARNING", logger="omop_llm.embeddings"):
            warn_if_prefixes_look_wrong(
                model="nomic-embed-text:v1.5",
                configuration={
                    "document_prefix": "search_document: ",
                    "query_prefix": "search_query: ",
                },
            )
        assert caplog.text == ""

    def test_warns_on_unrecognized_prefix(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level("WARNING", logger="omop_llm.embeddings"):
            warn_if_prefixes_look_wrong(
                model="some-new-model",
                configuration={"document_prefix": "totally_made_up: ", "query_prefix": "query: "},
            )
        assert "totally_made_up: " in caplog.text
        assert "doesn't match a commonly recognized" in caplog.text

    def test_does_not_raise_for_unrecognized_prefix(self) -> None:
        # A prefix outside KNOWN_EMBEDDING_PREFIXES is a heads-up, not an error.
        warn_if_prefixes_look_wrong(
            model="some-new-model", configuration={"document_prefix": "custom: ", "query_prefix": "custom: "}
        )

    def test_every_known_prefix_is_a_non_empty_string(self) -> None:
        for prefix in KNOWN_EMBEDDING_PREFIXES:
            assert isinstance(prefix, str)
            assert prefix
