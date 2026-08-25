"""Structured decision logging via Langfuse -- trace ID, decision path,
tokens/latency per action. Degrades gracefully to local-only logging if
Langfuse isn't configured (no credentials in this environment as of
2026-08-21) -- telemetry is an observability nice-to-have, not something
that should block generation/replay if the logging backend is
unavailable.
"""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def new_trace_id() -> str:
    return str(uuid.uuid4())


@dataclass
class DecisionRecord:
    """One entry in a decision path -- e.g. "probe scored, STOP chosen"
    or "budget exhausted mid-escalation, best-so-far returned."
    """

    trace_id: str
    stage: str
    action: str | None
    granted_tokens: int | None
    metadata: dict[str, Any] = field(default_factory=dict)


class TelemetryLogger:
    """Sends structured decision records to Langfuse if configured
    (LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY env vars present), else
    falls back to a local JSONL file so nothing is silently lost.
    """

    def __init__(self, local_fallback_path: str | Path = "notes/scratch/telemetry_local.jsonl"):
        self.local_fallback_path = Path(local_fallback_path)
        self._langfuse_client = None
        if os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY"):
            try:
                from langfuse import Langfuse  # heavy optional import

                self._langfuse_client = Langfuse()
            except Exception:
                # Never let telemetry-backend unavailability break the caller.
                self._langfuse_client = None

    @property
    def backend(self) -> str:
        return "langfuse" if self._langfuse_client is not None else "local_jsonl_fallback"

    def record(self, decision: DecisionRecord) -> None:
        if self._langfuse_client is not None:
            try:
                self._langfuse_client.trace(
                    id=decision.trace_id,
                    name=decision.stage,
                    metadata={"action": decision.action, "granted_tokens": decision.granted_tokens,
                              **decision.metadata},
                )
                return
            except Exception:
                pass  # fall through to local logging rather than lose the record entirely

        self.local_fallback_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.local_fallback_path, "a") as f:
            f.write(json.dumps({
                "trace_id": decision.trace_id,
                "stage": decision.stage,
                "action": decision.action,
                "granted_tokens": decision.granted_tokens,
                "metadata": decision.metadata,
            }) + "\n")
