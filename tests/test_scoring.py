"""
Day 7's `scoring/` module. Uses a fake in-memory PRMClient (no network)
so this runs in CI without credentials -- the real hosted PRM is
exercised separately, live, against real P1 pool data in
notes/2026-08-23.md before this was trusted.
"""

from __future__ import annotations

import tempfile

import pytest

from marginal_token.answers.taxonomy import FailureStatus
from marginal_token.backends.base import Provenance, Sample, now_iso
from marginal_token.pools.store import Pool, PoolManifest
from marginal_token.scoring.pipeline import PRMScoreStore, compute_score_id, score_pool, score_sample
from marginal_token.scoring.prm_client import PRMScoreResult
from marginal_token.scoring.segmentation import segment, segment_special_token, segment_step_prefix


class FakePRMClient:
    """Deterministic, in-memory stand-in for a real PRMClient. `fail_after`
    simulates a kill: the Nth call onward returns ok=False, so a
    scoring run against it produces a genuinely partial score set --
    exactly the scenario resumability needs to recover from.
    """

    role = "fake_prm"

    def __init__(self, fail_after: int | None = None):
        self.fail_after = fail_after
        self.calls = 0

    def score(self, query: str, steps: list[str]) -> PRMScoreResult:
        self.calls += 1
        if self.fail_after is not None and self.calls > self.fail_after:
            return PRMScoreResult(num_steps=0, step_rewards=[], mean_reward=float("nan"), ok=False,
                                    error="simulated kill")
        rewards = [0.9] * len(steps)
        return PRMScoreResult(num_steps=len(steps), step_rewards=rewards, mean_reward=0.9, ok=True)


def _sample(text: str, sample_idx: int) -> Sample:
    provenance = Provenance(
        model_id="fake-model", backend="api_host", provider="fake_provider",
        revision_or_api_model="fake-model-v1", temperature=0.8, top_p=0.95, max_tokens=1024,
        seed=None, seed_honored="unknown", logprobs_available=False, quantization="none",
        generated_at=now_iso(), pool_id="fake-pool-id",
    )
    return Sample(text=text, finish_reason="stop", completion_tokens=10, prompt_tokens=10,
                   provenance=provenance, problem_id="p0", sample_idx=sample_idx)


def _pool(samples: list[Sample]) -> Pool:
    manifest = PoolManifest(policy_ref="fake-policy", backend_ref="fake-backend",
                             benchmark_id="fake-bench", contract_key=None)
    pool = Pool(problem_id="p0", pool_id="fake-pool-id", manifest=manifest)
    for s in samples:
        pool.add(s)
    return pool


# --- segmentation -----------------------------------------------------


def test_segment_double_newline_splits_on_blank_lines():
    text = "First step.\n\nSecond step.\n\nThird step."
    assert segment(text, "double_newline") == ["First step.", "Second step.", "Third step."]


def test_segment_step_prefix_returns_empty_when_pattern_never_matches():
    text = "1.  **Calculate r:**\n    r = 3\n\n2.  **Calculate theta:**\n    theta = pi/2"
    assert segment_step_prefix(text) == []


def test_segment_step_prefix_splits_on_literal_step_marker():
    text = "Step 1: do this.\nStep 2: do that."
    assert segment_step_prefix(text) == ["do this.", "do that."]


def test_segment_special_token_none_always_returns_empty():
    assert segment_special_token("a<sep>b<sep>c", None) == []


def test_segment_special_token_splits_when_configured():
    assert segment_special_token("a<sep>b<sep>c", "<sep>") == ["a", "b", "c"]


def test_segment_unknown_convention_raises():
    with pytest.raises(ValueError):
        segment("text", "not_a_real_convention")


# --- score_sample: taxonomy-safe handling ------------------------------


def test_score_sample_ok_case_carries_full_per_step_array():
    sample = _sample("step one.\n\nstep two.", sample_idx=0)
    client = FakePRMClient()
    score = score_sample(sample, query="q", client=client, convention="double_newline")
    assert score.status == FailureStatus.OK.value
    assert score.num_steps == 2
    assert score.step_rewards == [0.9, 0.9]
    assert score.mean_reward == 0.9
    assert score.final_step_reward == 0.9


def test_score_sample_segmentation_failure_never_calls_the_prm():
    # double_newline can't structurally fail on non-empty text (worst
    # case it returns the whole text as one step) -- step_prefix is the
    # convention that genuinely returns [] when its marker never matches,
    # e.g. this policy's actual numbered-markdown style (see
    # segmentation.py's module docstring).
    sample = _sample("1.  **Calculate r:**\n    r = 3", sample_idx=0)
    client = FakePRMClient()
    score = score_sample(sample, query="q", client=client, convention="step_prefix")
    assert score.status == FailureStatus.STEP_SEGMENTATION_FAILED.value
    assert score.mean_reward is None
    assert client.calls == 0, "a segmentation failure must never reach the PRM as an empty-steps call"


def test_score_sample_prm_failure_is_recorded_not_silently_scored():
    sample = _sample("step one.\n\nstep two.", sample_idx=0)
    client = FakePRMClient(fail_after=0)  # every call fails
    score = score_sample(sample, query="q", client=client, convention="double_newline")
    assert score.status == FailureStatus.PRM_SCORE_MISSING.value
    assert score.mean_reward is None
    assert score.error is not None


# --- score_pool / PRMScoreStore: resumability --------------------------


def test_score_pool_resumes_after_a_kill_to_full_coverage():
    samples = [_sample(f"step a.\n\nstep b (sample {i}).", sample_idx=i) for i in range(5)]
    pool = _pool(samples)

    with tempfile.TemporaryDirectory() as tmp:
        store = PRMScoreStore(tmp)

        killed_client = FakePRMClient(fail_after=2)
        first = score_pool(pool, query="q", client=killed_client, store=store, convention="double_newline")
        assert len(first) == 5  # score_pool records a result (possibly a failure) for every sample
        n_failed_first = sum(1 for s in first if s.status == FailureStatus.PRM_SCORE_MISSING.value)
        assert n_failed_first == 3, "the simulated kill should show up as recorded PRM failures, not silent gaps"

        # A PRM failure is still "done" from score_pool's resumability
        # point of view (it's a recorded outcome, not a crash) -- calling
        # again with a healthy client should find nothing left to do.
        healthy_client = FakePRMClient()
        second = score_pool(pool, query="q", client=healthy_client, store=store, convention="double_newline")
        assert second == []
        assert healthy_client.calls == 0

        score_id = compute_score_id(pool.pool_id, "fake_prm", "double_newline")
        all_scores = store.load(score_id, pool.problem_id)
        assert len(all_scores) == 5
        assert sorted(s.sample_idx for s in all_scores) == list(range(5))


def test_different_convention_or_prm_role_never_collides_in_the_store():
    """Regression test for the exact bug class this module was built to
    avoid: re-scoring the same pool under a different segmentation
    convention (or a different PRM) must be treated as separate work,
    never silently skipped because a different combination already ran.
    """
    samples = [_sample("step a.\n\nstep b.", sample_idx=0)]
    pool = _pool(samples)

    with tempfile.TemporaryDirectory() as tmp:
        store = PRMScoreStore(tmp)
        client = FakePRMClient()

        first = score_pool(pool, query="q", client=client, store=store, convention="double_newline")
        assert len(first) == 1
        assert client.calls == 1

        # Same pool, same client role, DIFFERENT convention -- must NOT be
        # treated as already done just because double_newline already ran.
        second = score_pool(pool, query="q", client=client, store=store, convention="step_prefix")
        assert len(second) == 1
        assert client.calls == 1, "step_prefix finds no steps in this text -- segmentation failure, not a PRM call"
        assert second[0].status == FailureStatus.STEP_SEGMENTATION_FAILED.value

        id_a = compute_score_id(pool.pool_id, "fake_prm", "double_newline")
        id_b = compute_score_id(pool.pool_id, "fake_prm", "step_prefix")
        assert id_a != id_b
