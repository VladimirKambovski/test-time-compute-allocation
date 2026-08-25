"""Replays any controller over cached pools at zero inference cost.
Deliberately thin: this module must NOT contain any allocation logic of
its own (invariant #5) -- it only ever calls `controller.decide()` on
whatever Controller instance it's given, exactly like the gateway does.
If this file ever grows an if/else that chooses an action without
delegating to a Controller, that's the actual bug invariant #5 exists to
catch.
"""

from __future__ import annotations

from dataclasses import dataclass

from marginal_token.controller.base import Budget, Controller, Decision, Probe


@dataclass
class ReplayResult:
    problem_id: str
    decision: Decision


def replay_one(controller: Controller, problem_id: str, probe: Probe, budget: Budget) -> ReplayResult:
    """The entire replay engine, in one function: call decide() on the
    shared Controller, wrap the result with which problem it was for.
    No allocation logic lives here -- see module docstring.
    """
    decision = controller.decide(probe, budget)
    return ReplayResult(problem_id=problem_id, decision=decision)


def replay_many(controller: Controller, probes: dict[str, Probe], budget: Budget) -> list[ReplayResult]:
    return [replay_one(controller, pid, probe, budget) for pid, probe in probes.items()]
