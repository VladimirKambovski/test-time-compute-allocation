"""An OBVIOUSLY FAKE controller, for parity-testing gateway/replay
plumbing only. Hardcoded to always choose STOP, regardless of probe
content -- this is deliberate: it must not look like a real placeholder
policy that encodes any assumption about the pending H1/H4 question. The
real controller (oracle labels, logistic predictor) is held pending the
mentor conversation; this stub exists ONLY to prove replay and the
gateway call the same Controller object identically (invariant #5,
tests/test_controller_parity.py). It has no research meaning whatsoever
and must never be used for anything except that plumbing test.
"""

from __future__ import annotations

from marginal_token.controller.base import Budget, Decision, Probe


class AlwaysStopFakeController:
    """NOT a real policy. NOT a baseline. NOT an assumption about what
    STOP should be preferred. It exists solely so
    test_controller_parity.py has something deterministic to call
    through both the replay path and the gateway path and compare.
    """

    def featurize(self, probe: Probe) -> dict[str, float]:
        # Trivial, deterministic, and intentionally uninformative --
        # returning a fixed feature regardless of probe content, so
        # there is no way to mistake this for real feature engineering.
        return {"fake_stub_feature": 0.0}

    def decide(self, probe: Probe, budget: Budget) -> Decision:
        return Decision(
            action="stop",
            budget_grant=0,
            class_probs={"stop": 1.0, "sample": 0.0, "select": 0.0, "search": 0.0, "abstain": 0.0},
            rationale={"note": "AlwaysStopFakeController -- parity-test plumbing only, not a real policy"},
        )
