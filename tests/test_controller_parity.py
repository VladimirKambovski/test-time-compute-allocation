"""
THE most important test in the repo. Asserts that the offline replay
engine and the live gateway produce the SAME decision for the SAME
probe evidence, because they must call the identical Controller object.

Uses `AlwaysStopFakeController` -- an OBVIOUSLY fake, hardcoded-to-STOP
controller that exists only for this plumbing test. It has no research
meaning and must never be read as a real policy or an assumption about
the H1/H4 question (settled 2026-08-22/23 -- SELECT narrowed out, see
notes/2026-08-22.md/2026-08-23.md). The parity guarantee itself doesn't
depend on the controller's content -- it depends on replay and the
gateway both calling decide() on the same instance, which is exactly
what's checked here regardless of what that instance actually decides.

Day 10: the real controller now exists (`DetectiveController`) -- a
second test below re-runs the identical parity check against it, not
just the fake stub, since "does the real predictor's decide() actually
agree between replay and gateway" is a materially different question
from "does the plumbing itself work."
"""

from fastapi.testclient import TestClient

from marginal_token.controller import AlwaysStopFakeController, Budget, DetectiveController, Probe, featurize
from marginal_token.replay import replay_one
from marginal_token.backends.base import Provenance, Sample, now_iso
from marginal_token.controller.features import FEATURE_NAMES


def _fixed_probe(query: str) -> Probe:
    # Content is irrelevant -- AlwaysStopFakeController ignores it by
    # design. A fixed, deterministic probe fixture either way.
    return Probe(samples=[], features={"query": query})


def test_replay_and_gateway_agree_on_probe_decision():
    controller = AlwaysStopFakeController()  # ONE shared instance, per invariant #5
    budget = Budget(max_tokens=1024, max_latency_ms=None)
    query = "What is 2+2?"

    # --- replay path ---
    replay_result = replay_one(controller, problem_id="p1", probe=_fixed_probe(query), budget=budget)

    # --- gateway path ---
    from marginal_token.gateway import create_app

    app = create_app(controller=controller, probe_provider=_fixed_probe)
    client = TestClient(app)
    resp = client.post("/v1/solve", json={
        "query": query,
        "budget": {"max_tokens": budget.max_tokens, "max_latency_ms": budget.max_latency_ms},
        "policy": "detective",
        "trace": False,
    })
    assert resp.status_code == 200
    gateway_body = resp.json()

    # The actual parity assertion: same action, same budget_grant, same
    # class_probs -- computed by the SAME controller instance via two
    # different entry points.
    assert gateway_body["action"] == replay_result.decision.action
    assert gateway_body["decision_path"][0]["granted_tokens"] == replay_result.decision.budget_grant
    assert gateway_body["evidence"]["class_probs"] == replay_result.decision.class_probs

    # Sanity: it really is the fake stub deciding, not something silently
    # picking STOP for an unrelated reason -- both should be exactly
    # "stop" with the fake's exact fixed class_probs.
    assert replay_result.decision.action == "stop"
    assert replay_result.decision.class_probs == {
        "stop": 1.0, "sample": 0.0, "select": 0.0, "search": 0.0, "abstain": 0.0,
    }


def _fake_sample(text: str, sample_idx: int) -> Sample:
    provenance = Provenance(
        model_id="fake", backend="api_host", provider="fake", revision_or_api_model="fake",
        temperature=0.8, top_p=0.95, max_tokens=1024, seed=None, seed_honored="unknown",
        logprobs_available=False, quantization="none", generated_at=now_iso(), pool_id="p",
    )
    return Sample(text=text, finish_reason="stop", completion_tokens=5, prompt_tokens=5,
                   provenance=provenance, problem_id="p1", sample_idx=sample_idx)


def test_replay_and_gateway_agree_using_the_real_detective_controller():
    """Same parity check as above, but against the REAL predictor
    (Day 10) instead of the plumbing-only fake stub -- a materially
    different question: does a controller that actually computes
    features and calls a fitted model still produce byte-identical
    decisions through both entry points?
    """
    controller = DetectiveController()
    # Fast, synthetic, separable fit -- proving parity for the real
    # architecture doesn't require the full real P1 fit (that's
    # `notes/scratch/day10_fit_predictors.py`, run separately).
    x_stop = {**{f: 0.0 for f in FEATURE_NAMES}, "top1_vote_fraction": 1.0, "normalized_entropy": 0.0}
    x_sample = {**{f: 0.0 for f in FEATURE_NAMES}, "top1_vote_fraction": 0.2, "normalized_entropy": 0.9}
    controller.fit([x_stop, x_sample] * 5, ["stop", "sample"] * 5)

    budget = Budget(max_tokens=1024, max_latency_ms=None)
    probe = Probe(samples=[_fake_sample("\\boxed{4}", i) for i in range(4)])  # unanimous -> should predict "stop"

    replay_result = replay_one(controller, problem_id="p1", probe=probe, budget=budget)

    from marginal_token.gateway import create_app

    def _probe_provider(query: str) -> Probe:
        return probe  # identical probe regardless of query text -- this test is about parity, not routing

    app = create_app(controller=controller, probe_provider=_probe_provider)
    client = TestClient(app)
    resp = client.post("/v1/solve", json={
        "query": "What is 2+2?",
        "budget": {"max_tokens": budget.max_tokens, "max_latency_ms": budget.max_latency_ms},
        "policy": "detective",
        "trace": False,
    })
    assert resp.status_code == 200
    gateway_body = resp.json()

    assert gateway_body["action"] == replay_result.decision.action
    assert gateway_body["decision_path"][0]["granted_tokens"] == replay_result.decision.budget_grant
    assert gateway_body["evidence"]["class_probs"] == replay_result.decision.class_probs
    assert replay_result.decision.action == "stop"  # the unanimous probe should genuinely predict this
