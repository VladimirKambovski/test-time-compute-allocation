"""Backend protocol, decode config, and the provenance block every sample
must carry. See CLAUDE.md invariant #3 and docs/brief.md §27.1/§27.6.

One interface (`Backend`), multiple implementations (local vLLM, hosted
API, Bedrock) -- callers never branch on backend type, they call
`generate()` and read `capabilities()` to know what's available (e.g.
whether logprobs are exposed).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

BackendType = Literal["local_vllm", "api_host", "bedrock"]


@dataclass(frozen=True)
class DecodeConfig:
    """Everything that's part of pool identity per docs/brief.md §27.6's
    compatibility contract, plus max_tokens (also contract-relevant --
    changes the effective output-length distribution).
    """

    temperature: float
    top_p: float
    max_tokens: int
    seed: int | None = None
    thinking_mode: bool = False
    stop_sequences: tuple[str, ...] = ()


@dataclass(frozen=True)
class BackendCaps:
    """What a backend can actually do -- callers (e.g. the predictor's
    feature extraction, §16) degrade gracefully rather than assume.
    """

    logprobs_available: bool
    seed_honored: Literal[True, False, "unknown"]
    max_concurrency_hint: int | None = None


@dataclass
class Provenance:
    """The exact block docs/brief.md §27.1 requires on every sample.
    `tests/test_backend_metadata.py` fails any sample missing this.
    """

    model_id: str
    backend: BackendType
    provider: str
    revision_or_api_model: str
    temperature: float
    top_p: float
    max_tokens: int
    seed: int | None
    seed_honored: Literal[True, False, "unknown"]
    logprobs_available: bool
    quantization: str
    generated_at: str
    pool_id: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "backend": self.backend,
            "provider": self.provider,
            "revision_or_api_model": self.revision_or_api_model,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "seed": self.seed,
            "seed_honored": self.seed_honored,
            "logprobs_available": self.logprobs_available,
            "quantization": self.quantization,
            "generated_at": self.generated_at,
            "pool_id": self.pool_id,
        }

    # §27.6's compatibility contract: these fields must be identical for
    # two samples to belong to the same pool. Deliberately excludes
    # seed/seed_honored/generated_at/logprobs_available, which are
    # "recorded but permitted to differ within one pool" per §27.6.
    CONTRACT_FIELDS = (
        "model_id",
        "backend",
        "provider",
        "revision_or_api_model",
        "quantization",
        "temperature",
        "top_p",
        "max_tokens",
    )

    def contract_key(self) -> tuple:
        d = self.as_dict()
        return tuple(d[f] for f in self.CONTRACT_FIELDS)


REQUIRED_PROVENANCE_FIELDS = (
    "model_id",
    "backend",
    "provider",
    "revision_or_api_model",
    "temperature",
    "top_p",
    "max_tokens",
    "seed",
    "seed_honored",
    "logprobs_available",
    "quantization",
    "generated_at",
    "pool_id",
)


def validate_provenance(provenance: dict[str, Any]) -> None:
    """Raise if `provenance` is missing any required field. Never silently
    accept a partial provenance block -- see invariant #3.
    """
    missing = [f for f in REQUIRED_PROVENANCE_FIELDS if f not in provenance]
    if missing:
        raise ValueError(f"Sample provenance missing required field(s): {missing}. "
                          f"Every sample must carry full provenance (CLAUDE.md invariant #3).")


@dataclass
class Sample:
    """One generated completion, with its full provenance block."""

    text: str
    finish_reason: str
    completion_tokens: int
    prompt_tokens: int
    provenance: Provenance
    logprobs: list[dict[str, Any]] | None = None
    problem_id: str = ""
    sample_idx: int = 0
    extra: dict[str, Any] = field(default_factory=dict)


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


class Backend(Protocol):
    """One interface, multiple implementations. Callers never branch on
    backend type -- see docs/brief.md §22.
    """

    def generate(self, prompts: list[str], cfg: DecodeConfig) -> list[Sample]: ...

    def capabilities(self) -> BackendCaps: ...
