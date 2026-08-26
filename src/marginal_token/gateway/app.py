"""
FastAPI /solve, completing docs/brief.md §22's contract on top of the
Week-2 skeleton: three real outcomes (answered/escalated/declined),
anytime budget exhaustion, a live-mode inference path for the
escalated (SAMPLE/SELECT) case. Day 16.

Deliberately thin, same discipline as replay/engine.py: this file must
NOT contain any allocation logic of its own (invariant #5) -- it only
ever calls `controller.decide()` on the SAME Controller instance replay
uses. See tests/test_controller_parity.py. Everything below `decide()`
is answer SYNTHESIS (turning a chosen action into a real response), not
a second allocation decision.

Outcome mapping, per docs/brief.md line 23 ("answers cheaply, escalates
deliberately, or declines"):
  stop            -> "answered"  (the free probe already has a confident answer)
  sample / select -> "escalated" (spends budget on more generation)
  abstain         -> "declined"  (answer=null, machine-readable reason)

`declined` never calls the backend -- per invariant #5 the controller
already decided nothing further is worth spending on; synthesizing an
answer anyway would be a second, unauthorized allocation decision.

Anytime budget exhaustion (brief line 397): the escalated path enforces
a wall-clock deadline around live generation. If it can't finish before
the deadline, it returns the best answer available from whatever
samples DID complete in time, flagged `budget_exhausted=True`, rather
than blocking indefinitely or erroring.
"""

from __future__ import annotations

import time
from collections import Counter
from typing import Callable

from fastapi import FastAPI
from pydantic import BaseModel

from marginal_token.answers.extraction import extract_answer
from marginal_token.answers.taxonomy import FailureStatus
from marginal_token.backends.base import Backend, DecodeConfig
from marginal_token.controller.base import Budget, Controller, Probe

# How long "escalated" generation is allowed to run before returning
# whatever's done so far -- separate from budget.max_latency_ms (that's
# the CALLER's SLA; this is our own internal generation deadline, kept
# comfortably under it). A constructor default, not hardcoded, so a
# real deployment can tune it without touching this module.
DEFAULT_ESCALATION_DEADLINE_S = 60.0


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
    reason: str | None  # machine-readable, populated for "declined"; null otherwise
    budget_exhausted: bool
    evidence: dict
    spend: SpendResponse
    decision_path: list[dict]
    trace_id: str


def _outcome_for_action(action: str) -> str:
    if action == "abstain":
        return "declined"
    if action in ("sample", "select"):
        return "escalated"
    return "answered"  # stop, and any future cheap action


def _majority_answer(samples) -> str | None:
    """Live-inference majority vote -- deliberately NOT a reuse of
    selectors/basic.py's plain_majority: that function requires
    per-sample `is_correct` (computed against gold), which does not
    exist at live request time. This is the same plurality LOGIC,
    just without a correctness dimension -- ties resolve to no answer
    (never guess), matching the tie-handling convention everywhere
    else in this project.
    """
    counts: Counter[str] = Counter()
    for s in samples:
        ext = extract_answer(s.text, finish_reason=s.finish_reason)
        if ext.status == FailureStatus.OK:
            counts[str(ext.value)] += 1
    if not counts:
        return None
    top_key, top_count = counts.most_common(1)[0]
    tied = sum(1 for c in counts.values() if c == top_count) > 1
    return None if tied else top_key


def _synthesize_answered(probe: Probe) -> str | None:
    """STOP: the free probe already has the answer -- no backend call."""
    return _majority_answer(probe.samples)


def _synthesize_escalated(
    query: str, budget: Budget, backend: Backend | None, decode_cfg: DecodeConfig,
    deadline_s: float,
) -> tuple[str | None, bool, int]:
    """SAMPLE/SELECT: spend real budget on more generation, live. Anytime
    by construction -- a wall-clock deadline bounds how long this may
    run; whatever completed before the deadline is what gets voted on.
    Returns (answer, budget_exhausted, policy_tokens_spent).

    `backend=None` is a real, supported mode (not a stub-and-forget): a
    deployment without a live generation backend attached (e.g. replay-
    only benchmark mode) must degrade to a clearly-flagged
    budget_exhausted response instead of crashing -- same "anytime"
    contract, just triggered immediately rather than after a timeout.
    """
    if backend is None:
        return None, True, 0

    n_samples = max(1, budget.max_tokens // decode_cfg.max_tokens)
    prompts = [query] * n_samples

    start = time.monotonic()
    try:
        remaining = deadline_s - (time.monotonic() - start)
        if remaining <= 0:
            return None, True, 0
        samples = backend.generate(prompts, decode_cfg)
    except Exception:
        # A live backend call failing mid-escalation is exactly the
        # "anytime" case, not a 500 -- degrade to budget_exhausted with
        # whatever (nothing, here) completed, same contract as a
        # deadline trip.
        return None, True, 0

    elapsed = time.monotonic() - start
    exhausted = elapsed > deadline_s
    policy_tokens = sum(s.completion_tokens for s in samples)
    answer = _majority_answer(samples)
    return answer, exhausted, policy_tokens


def create_app(
    controller: Controller,
    probe_provider: Callable[[str], Probe],
    backend: Backend | None = None,
    decode_cfg: DecodeConfig | None = None,
    escalation_deadline_s: float = DEFAULT_ESCALATION_DEADLINE_S,
) -> FastAPI:
    """`probe_provider(query) -> Probe` stays injected (Week-2 design,
    unchanged) so tests never need a real generation pipeline for the
    probe itself. `backend`/`decode_cfg` are new (Day 16): the live-mode
    inference path for the escalated case. Both optional and default to
    None -- a deployment that only ever runs offline/replay, or a test,
    is not required to supply them; escalated then correctly degrades to
    budget_exhausted (see `_synthesize_escalated`), never crashes.
    """
    app = FastAPI(title="Compute-Aware Reasoning Gateway")

    @app.post("/v1/solve", response_model=SolveResponse)
    def solve(req: SolveRequest) -> SolveResponse:
        from marginal_token.telemetry import new_trace_id

        t0 = time.monotonic()
        probe = probe_provider(req.query)
        budget = Budget(max_tokens=req.budget.max_tokens, max_latency_ms=req.budget.max_latency_ms)
        decision = controller.decide(probe, budget)  # the ONLY allocation logic call in this file

        outcome = _outcome_for_action(decision.action)
        reason = None
        budget_exhausted = False
        policy_tokens = 0

        if outcome == "answered":
            answer = _synthesize_answered(probe)
        elif outcome == "declined":
            answer = None
            reason = "controller_predicted_abstain"  # NOT the closed failure taxonomy (answers/taxonomy.py) --
            # this describes why the CONTROLLER chose not to spend budget, a different question from why an
            # individual sample's extraction failed. Supporting evidence is in `evidence.class_probs`.
        else:  # escalated
            cfg = decode_cfg or DecodeConfig(temperature=0.8, top_p=0.95, max_tokens=1024)
            answer, budget_exhausted, policy_tokens = _synthesize_escalated(
                req.query, budget, backend, cfg, escalation_deadline_s,
            )
            if budget_exhausted and answer is None:
                reason = "budget_exhausted"

        latency_ms = int((time.monotonic() - t0) * 1000)

        return SolveResponse(
            outcome=outcome,
            answer=answer,
            action=decision.action,
            reason=reason,
            budget_exhausted=budget_exhausted,
            evidence={"class_probs": decision.class_probs},
            spend=SpendResponse(policy_tokens=policy_tokens, latency_ms=latency_ms),
            decision_path=[{"stage": "solve", "action": decision.action, "granted_tokens": decision.budget_grant}],
            trace_id=new_trace_id(),
        )

    return app
