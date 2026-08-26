"""
Day 16: docs/brief.md §22's full /solve contract -- three real outcomes,
anytime budget exhaustion, a live-mode path for escalated. Roadmap's own
"done when" bar: a `declined` response produced end to end with a
machine-readable reason, parity green. Covers all three outcomes, not
just declined, since `_outcome_for_action`/answer synthesis are new
logic this session and deserve direct coverage, not just an inference
from the parity tests.

Same discipline as test_controller_parity.py: obviously-fake,
single-purpose controllers, never real policies, so these tests can't
be mistaken for encoding an assumption about H1/H4.
"""

from fastapi.testclient import TestClient

from marginal_token.answers.taxonomy import FailureStatus
from marginal_token.backends.base import Provenance, Sample, now_iso
from marginal_token.controller.base import Budget, Decision, Probe
from marginal_token.gateway import create_app


class _AlwaysAbstainFakeController:
    """NOT a real policy -- parity-test plumbing only, mirrors
    AlwaysStopFakeController's exact discipline (stub.py) for the one
    other action this test file needs deterministically.
    """

    def featurize(self, probe: Probe) -> dict[str, float]:
        return {"fake_stub_feature": 0.0}

    def decide(self, probe: Probe, budget: Budget) -> Decision:
        return Decision(
            action="abstain", budget_grant=0,
            class_probs={"stop": 0.0, "sample": 0.0, "select": 0.0, "search": 0.0, "abstain": 1.0},
            rationale={"note": "AlwaysAbstainFakeController -- test plumbing only"},
        )


class _AlwaysSampleFakeController:
    """NOT a real policy -- forces the escalated path deterministically."""

    def featurize(self, probe: Probe) -> dict[str, float]:
        return {"fake_stub_feature": 0.0}

    def decide(self, probe: Probe, budget: Budget) -> Decision:
        return Decision(
            action="sample", budget_grant=budget.max_tokens,
            class_probs={"stop": 0.0, "sample": 1.0, "select": 0.0, "search": 0.0, "abstain": 0.0},
            rationale={"note": "AlwaysSampleFakeController -- test plumbing only"},
        )


def _fake_sample(text: str, sample_idx: int) -> Sample:
    provenance = Provenance(
        model_id="fake", backend="api_host", provider="fake", revision_or_api_model="fake",
        temperature=0.8, top_p=0.95, max_tokens=1024, seed=None, seed_honored="unknown",
        logprobs_available=False, quantization="none", generated_at=now_iso(), pool_id="p",
    )
    return Sample(text=text, finish_reason="stop", completion_tokens=5, prompt_tokens=5,
                   provenance=provenance, problem_id="p1", sample_idx=sample_idx)


def _empty_probe(query: str) -> Probe:
    return Probe(samples=[])


def test_declined_outcome_has_machine_readable_reason_and_no_answer():
    """The roadmap's own 'done when' bar for Day 16."""
    app = create_app(controller=_AlwaysAbstainFakeController(), probe_provider=_empty_probe)
    client = TestClient(app)
    resp = client.post("/v1/solve", json={
        "query": "an unsolvable problem",
        "budget": {"max_tokens": 1024, "max_latency_ms": None},
        "policy": "detective",
        "trace": False,
    })
    assert resp.status_code == 200
    body = resp.json()
    assert body["outcome"] == "declined"
    assert body["answer"] is None
    assert body["reason"] == "controller_predicted_abstain"
    assert isinstance(body["reason"], str) and body["reason"]  # genuinely machine-readable, not empty/null
    assert body["action"] == "abstain"


def test_answered_outcome_synthesizes_the_probe_majority():
    """STOP: answer comes from the free probe alone, no backend call."""
    from marginal_token.controller.stub import AlwaysStopFakeController

    probe_samples = [_fake_sample("\\boxed{42}", i) for i in range(4)]

    def probe_provider(query: str) -> Probe:
        return Probe(samples=probe_samples)

    app = create_app(controller=AlwaysStopFakeController(), probe_provider=probe_provider)
    client = TestClient(app)
    resp = client.post("/v1/solve", json={
        "query": "what is 6*7", "budget": {"max_tokens": 1024, "max_latency_ms": None},
        "policy": "detective", "trace": False,
    })
    body = resp.json()
    assert body["outcome"] == "answered"
    assert body["answer"] == "42"
    assert body["reason"] is None
    assert body["budget_exhausted"] is False


def test_escalated_outcome_without_a_backend_degrades_to_budget_exhausted():
    """Anytime by construction (brief line 397): no backend attached is
    a real, supported deployment mode (e.g. replay-only benchmark mode),
    not a crash -- degrades to budget_exhausted immediately, same
    contract as a deadline trip mid-generation.
    """
    app = create_app(controller=_AlwaysSampleFakeController(), probe_provider=_empty_probe, backend=None)
    client = TestClient(app)
    resp = client.post("/v1/solve", json={
        "query": "needs more sampling", "budget": {"max_tokens": 4096, "max_latency_ms": None},
        "policy": "detective", "trace": False,
    })
    body = resp.json()
    assert body["outcome"] == "escalated"
    assert body["answer"] is None
    assert body["budget_exhausted"] is True
    assert body["reason"] == "budget_exhausted"
    assert body["action"] == "sample"


def test_escalated_outcome_with_a_working_backend_returns_a_real_answer():
    """The live-mode inference path, Day 16's other headline item."""
    class _FakeBackend:
        def generate(self, prompts, cfg):
            return [_fake_sample("\\boxed{7}", i) for i in range(len(prompts))]

    app = create_app(
        controller=_AlwaysSampleFakeController(), probe_provider=_empty_probe,
        backend=_FakeBackend(), decode_cfg=None,
    )
    client = TestClient(app)
    resp = client.post("/v1/solve", json={
        "query": "needs more sampling", "budget": {"max_tokens": 2048, "max_latency_ms": None},
        "policy": "detective", "trace": False,
    })
    body = resp.json()
    assert body["outcome"] == "escalated"
    assert body["answer"] == "7"
    assert body["budget_exhausted"] is False
    assert body["reason"] is None
    assert body["spend"]["policy_tokens"] > 0
