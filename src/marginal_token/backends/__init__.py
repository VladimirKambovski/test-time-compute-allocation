"""Pluggable generation backends (local vLLM, hosted API, Bedrock). One interface, full provenance metadata on every sample. See CLAUDE.md invariant 3."""

from marginal_token.backends.base import (
    Backend,
    BackendCaps,
    DecodeConfig,
    Provenance,
    REQUIRED_PROVENANCE_FIELDS,
    Sample,
    validate_provenance,
)

__all__ = [
    "REQUIRED_PROVENANCE_FIELDS",
    "Backend",
    "BackendCaps",
    "DecodeConfig",
    "Provenance",
    "Sample",
    "validate_provenance",
]
