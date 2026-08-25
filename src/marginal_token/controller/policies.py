"""
The 5 simple fixed policies from docs/brief.md line 276 (E7's
Pareto-frontier comparison, `Miser`, `Spendthrift`, `Uniform-Select`,
`Gambler`, `Oracle` -- `Fortune Teller` and `Detective` are the two
learned ones, in `predictor.py`, since they need fitting). Built as
working scaffolds per Day 10's roadmap item; E7 itself (real numbers,
real budget calibration) is Day 15's job.

Every class implements the shared `Controller` protocol so replay/
gateway never need to know which policy they're calling (invariant #5).
"""

from __future__ import annotations

import random

from marginal_token.controller.base import Budget, Decision, Probe

_ALL_ACTIONS = ("stop", "sample", "select", "search", "abstain")


def _full_class_probs(winner: str, prob: float = 1.0) -> dict[str, float]:
    return {a: (prob if a == winner else 0.0) for a in _ALL_ACTIONS}


class MiserController:
    """A0 always -- the cheapest possible policy, never spends beyond
    the free probe.
    """

    def featurize(self, probe: Probe) -> dict[str, float]:
        return {}

    def decide(self, probe: Probe, budget: Budget) -> Decision:
        return Decision(action="stop", budget_grant=0, class_probs=_full_class_probs("stop"),
                          rationale={"policy": "miser"})


class SpendthriftController:
    """A1 at max budget -- always spends the full grant on more
    sampling, regardless of what the probe evidence shows.
    """

    def featurize(self, probe: Probe) -> dict[str, float]:
        return {}

    def decide(self, probe: Probe, budget: Budget) -> Decision:
        return Decision(action="sample", budget_grant=budget.max_tokens, class_probs=_full_class_probs("sample"),
                          rationale={"policy": "spendthrift"})


class UniformSelectController:
    """A2 always -- a fixed COMPARATOR baseline for E7, not a reversal
    of the SELECT-narrowing decision. That decision is about the
    LEARNED controller's (Detective's) action space
    (notes/2026-08-22.md/2026-08-23.md); whether SELECT is worth
    comparing against as a fixed baseline in the Pareto-frontier
    analysis is a separate question docs/brief.md's own E7 design
    already answers "yes" to (it's literally named in the fixed-policy
    list), independent of the controller-narrowing decision.
    """

    def featurize(self, probe: Probe) -> dict[str, float]:
        return {}

    def decide(self, probe: Probe, budget: Budget) -> Decision:
        return Decision(action="select", budget_grant=budget.max_tokens, class_probs=_full_class_probs("select"),
                          rationale={"policy": "uniform_select"})


class GamblerController:
    """Random action at a matched rate. docs/brief.md line 276 doesn't
    pin down what "matched" means numerically -- taken here as a
    constructor parameter (the STOP probability) rather than a
    hardcoded guess, since the real calibration (matching Detective's
    own marginal STOP rate at a given budget, most likely) is Day 15's
    (E7's) job, not something to decide unilaterally today. Deterministic
    under a given seed, so a replay run is reproducible.
    """

    def __init__(self, stop_probability: float = 0.5, seed: int = 0):
        if not 0.0 <= stop_probability <= 1.0:
            raise ValueError(f"stop_probability must be in [0,1], got {stop_probability}")
        self.stop_probability = stop_probability
        self._rng = random.Random(seed)

    def featurize(self, probe: Probe) -> dict[str, float]:
        return {}

    def decide(self, probe: Probe, budget: Budget) -> Decision:
        action = "stop" if self._rng.random() < self.stop_probability else "sample"
        grant = 0 if action == "stop" else budget.max_tokens
        probs = _full_class_probs("stop", self.stop_probability)
        probs["sample"] = 1.0 - self.stop_probability
        return Decision(action=action, budget_grant=grant, class_probs=probs,
                          rationale={"policy": "gambler", "stop_probability": self.stop_probability})


class OracleController:
    """The ceiling: chooses the TRUE best action per problem, given
    ground truth. Explicitly cheating (a reference ceiling, not a real
    policy) -- usable only in offline replay against a real pool, and
    only when the oracle label has already been computed
    (`oracle_labels.oracle_action_label`) and attached to
    `probe.features["oracle_action"]` by the CALLER. This class
    deliberately does not recompute it from gold internally, so it can
    never be accidentally wired into a live-inference path that
    shouldn't have access to gold answers at decide() time.
    """

    def featurize(self, probe: Probe) -> dict[str, float]:
        return probe.features

    def decide(self, probe: Probe, budget: Budget) -> Decision:
        action = probe.features.get("oracle_action")
        if action not in ("stop", "sample", "select", "abstain"):
            raise ValueError(
                "OracleController requires probe.features['oracle_action'] to already be one of "
                "'stop'/'sample'/'select'/'abstain' (offline replay only) -- see oracle_labels.py. "
                f"Got: {action!r}"
            )
        # SAMPLE and SELECT get the SAME nominal budget B (CLAUDE.md
        # invariant: SELECT spends part of B on PRM scoring, buying
        # FEWER raw samples at equal budget -- that split is
        # budget/accounting.py's job, not a difference in the grant
        # amount itself).
        grant = {"stop": 0, "sample": budget.max_tokens, "select": budget.max_tokens, "abstain": 0}[action]
        return Decision(action=action, budget_grant=grant, class_probs=_full_class_probs(action),
                          rationale={"policy": "oracle"})
