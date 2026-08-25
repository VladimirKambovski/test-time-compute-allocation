"""
Day 6's own "done when" condition: a killed generation run resumes
cleanly to a complete pool, and every sample carries full provenance.

Uses a fake in-process Backend (no network) so this runs in CI without
credentials -- the real hosted backend is exercised separately, live,
in notes/2026-08-23.md's Day 6 smoke test before any real P1 generation
was started.
"""

from __future__ import annotations

import tempfile

from marginal_token.backends.base import BackendCaps, DecodeConfig, Provenance, Sample, now_iso, validate_provenance
from marginal_token.generation.sweep import SweepTask, run_sweep
from marginal_token.pools.store import PoolStore, compute_pool_id


class FakeBackend:
    """Deterministic, in-memory stand-in for a real Backend. `fail_after`
    simulates a kill: the Nth call onward raises, so a sweep run against
    it produces a genuinely partial pool -- exactly the scenario
    resumability needs to recover from.
    """

    backend_type = "api_host"
    provider = "fake_test_provider"
    revision_or_api_model = "fake-model-v1"
    quantization = "none"

    def __init__(self, fail_after: int | None = None):
        self.fail_after = fail_after
        self.calls = 0

    def capabilities(self) -> BackendCaps:
        return BackendCaps(logprobs_available=False, seed_honored="unknown", max_concurrency_hint=4)

    def generate(self, prompts: list[str], cfg: DecodeConfig) -> list[Sample]:
        out = []
        for prompt in prompts:
            self.calls += 1
            if self.fail_after is not None and self.calls > self.fail_after:
                raise RuntimeError("simulated kill")
            provenance = Provenance(
                model_id=self.revision_or_api_model,
                backend=self.backend_type,
                provider=self.provider,
                revision_or_api_model=self.revision_or_api_model,
                temperature=cfg.temperature,
                top_p=cfg.top_p,
                max_tokens=cfg.max_tokens,
                seed=cfg.seed,
                seed_honored="unknown",
                logprobs_available=False,
                quantization=self.quantization,
                generated_at=now_iso(),
                pool_id="",
            )
            out.append(Sample(text=f"\\boxed{{{prompt}}}", finish_reason="stop",
                                completion_tokens=5, prompt_tokens=5, provenance=provenance))
        return out


def _cfg() -> DecodeConfig:
    return DecodeConfig(temperature=0.8, top_p=0.95, max_tokens=1024, seed=None)


def test_run_sweep_resumes_after_a_kill_to_a_complete_pool():
    tasks = [SweepTask(problem_id=f"p{i}", prompt=f"prompt-{i}") for i in range(3)]
    n = 4  # small N for a fast test -- the mechanism is N-independent

    with tempfile.TemporaryDirectory() as tmp:
        store = PoolStore(tmp)

        # First run: killed partway through (fail_after < total tasks).
        killed_backend = FakeBackend(fail_after=5)  # 3 problems x 4 samples = 12 total; kill at 5
        first = run_sweep(
            tasks=tasks, n=n, backend=killed_backend, cfg=_cfg(), store=store,
            policy_ref="fake-policy", backend_ref="fake-backend", benchmark_id="fake-bench",
            max_workers=1,  # deterministic call ordering for a clean partial-completion assertion
        )
        assert first.completed < 12, "the simulated kill should have left the pool incomplete"
        assert first.failed, "the calls past fail_after should show up as failures, not be silently dropped"

        # Resume: a fresh, non-failing backend picks up exactly where the
        # killed run left off -- `run_sweep` re-reads `done_sample_indices`
        # fresh each call, so this is a real resume, not a mocked one.
        healthy_backend = FakeBackend(fail_after=None)
        second = run_sweep(
            tasks=tasks, n=n, backend=healthy_backend, cfg=_cfg(), store=store,
            policy_ref="fake-policy", backend_ref="fake-backend", benchmark_id="fake-bench",
            max_workers=1,
        )
        assert first.completed + second.completed == 12, "resumed run must fill in exactly the missing samples"

        # The complete pool: every (problem, sample_idx) present exactly
        # once, and every sample carries full provenance.
        for task in tasks:
            pool_id = compute_pool_id("fake-policy", "fake-backend", "fake-bench", task.problem_id,
                                        0.8, 0.95, 1024, None, n)
            pool = store.load(pool_id, task.problem_id, "fake-bench", "fake-policy", "fake-backend")
            assert len(pool) == n, f"{task.problem_id}: expected {n} samples, got {len(pool)}"
            sample_indices = sorted(s.sample_idx for s in pool.samples)
            assert sample_indices == list(range(n)), "no duplicates, no gaps after resume"
            for sample in pool.samples:
                validate_provenance(sample.provenance.as_dict())  # raises if anything is missing


def test_run_sweep_is_a_no_op_on_an_already_complete_pool():
    """Calling run_sweep again on a pool that's already full must not
    re-generate anything -- the whole point of checking
    `done_sample_indices` fresh each call.
    """
    tasks = [SweepTask(problem_id="only", prompt="prompt")]
    n = 2

    with tempfile.TemporaryDirectory() as tmp:
        store = PoolStore(tmp)
        backend = FakeBackend()
        run_sweep(tasks=tasks, n=n, backend=backend, cfg=_cfg(), store=store,
                   policy_ref="p", backend_ref="b", benchmark_id="bench", max_workers=1)
        assert backend.calls == n

        result = run_sweep(tasks=tasks, n=n, backend=backend, cfg=_cfg(), store=store,
                             policy_ref="p", backend_ref="b", benchmark_id="bench", max_workers=1)
        assert result.completed == 0
        assert backend.calls == n  # unchanged -- nothing new was generated
