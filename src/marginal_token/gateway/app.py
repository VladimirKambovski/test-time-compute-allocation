"""Minimal FastAPI /solve skeleton, per docs/brief.md §22's API contract.
No UI, no real probe-generation pipeline wired in yet (that needs
backends + answers/ extraction + real feature engineering, out of scope
for this skeleton) -- a `probe_provider` callable is injected instead,
so tests (and later, the real pipeline) can supply a Probe without this
module needing to know how one gets built.

Deliberately thin, same discipline as replay/engine.py: this file must
NOT contain any allocation logic of its own (invariant #5) -- it only
ever calls `controller.decide()` on the SAME Controller instance replay
uses. See tests/test_controller_parity.py.
"""

from __future__ import annotations

from typing import Callable

from fastapi import FastAPI
from pydantic import BaseModel

from marginal_token.controller.base import Budget, Controller, Probe


class BudgetRequest(BaseModel):
    max_tokens: int
    max_latency_ms: int | None = None


class SolveRequest(BaseModel):
    query: str
    budget: BudgetRequest
    policy: str = "detective"
    trace: bool = False


class SpendResponse(BaseModel):
    policy_tokens: int = 0
    prm_forwards: int = 0
    discarded_beam_tokens: int = 0
    latency_ms: int = 0
    usd_equivalent: float = 0.0


class SolveResponse(BaseModel):
    outcome: str  # "answered" | "escalated" | "declined"
    answer: str | None
    action: str
    evidence: dict
    spend: SpendResponse
    decision_path: list[dict]
    trace_id: str


def _outcome_for_action(action: str) -> str:
    if action == "abstain":
        return "declined"
    return "answered"


def create_app(controller: Controller, probe_provider: Callable[[str], Probe]) -> FastAPI:
    """`probe_provider(query) -> Probe` is injected so this skeleton
    doesn't need the real generation/extraction/feature pipeline to
    exist yet -- exactly the kind of substitution that would be
    impossible if allocation logic lived here instead of in the shared
    Controller.
    """
    app = FastAPI(title="Compute-Aware Reasoning Gateway (skeleton)")

    @app.post("/v1/solve", response_model=SolveResponse)
    def solve(req: SolveRequest) -> SolveResponse:
        from marginal_token.telemetry import new_trace_id

        probe = probe_provider(req.query)
        budget = Budget(max_tokens=req.budget.max_tokens, max_latency_ms=req.budget.max_latency_ms)
        decision = controller.decide(probe, budget)  # the ONLY allocation logic call in this file

        return SolveResponse(
            outcome=_outcome_for_action(decision.action),
            answer=None,  # real answer synthesis is out of scope for this skeleton
            action=decision.action,
            evidence={"class_probs": decision.class_probs},
            spend=SpendResponse(),  # real spend tracking wires in budget/ once generation is attached
            decision_path=[{"stage": "solve", "action": decision.action, "granted_tokens": decision.budget_grant}],
            trace_id=new_trace_id(),
        )

    return app
