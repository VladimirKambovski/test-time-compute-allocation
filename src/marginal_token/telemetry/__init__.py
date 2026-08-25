"""Structured decision logging via Langfuse -- trace ID, decision path, tokens/latency per action."""

from marginal_token.telemetry.logger import DecisionRecord, TelemetryLogger, new_trace_id

__all__ = ["DecisionRecord", "TelemetryLogger", "new_trace_id"]
