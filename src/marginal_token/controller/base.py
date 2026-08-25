"""Controller protocol and its input/output types, per docs/brief.md §22.
The real controller (featurize -> oracle labels -> logistic predictor)
is explicitly NOT built here -- that's held pending the mentor
conversation about the H1/H4 framing (see notes/2026-08-21.md). This
module only defines the shared interface that a real controller,
replay, and the gateway all agree on.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from marginal_token.backends.base import Sample

Action = Literal["stop", "sample", "select", "search", "abstain"]


@dataclass
class Probe:
    samples: list[Sample]
    features: dict[str, float] = field(default_factory=dict)


@dataclass
class Budget:
    max_tokens: int
    max_latency_ms: int | None = None


@dataclass
class Decision:
    action: Action
    budget_grant: int
    class_probs: dict[str, float] = field(default_factory=dict)
    rationale: dict[str, Any] = field(default_factory=dict)


class Controller(Protocol):
    """One object, consumed IDENTICALLY by replay and the gateway --
    invariant #5. Neither replay nor gateway may contain allocation
    logic of their own; they only ever call decide() on one shared
    instance.
    """

    def featurize(self, probe: Probe) -> dict[str, float]: ...

    def decide(self, probe: Probe, budget: Budget) -> Decision: ...
