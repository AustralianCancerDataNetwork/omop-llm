"""Exceptions raised by omop_llm."""

from __future__ import annotations


class OmopLlmError(RuntimeError):
    """Base class for all omop_llm errors."""


class UnsupportedProviderError(OmopLlmError):
    """Raised when a provider key is not in omop_llm's supported registry."""


class UnsupportedCapabilityError(OmopLlmError):
    """Raised when a requested capability is not available on the resolved backend."""


class NoParsedOutputError(OmopLlmError):
    """Raised when a structured-output call produced no parsed instance to unwrap."""
