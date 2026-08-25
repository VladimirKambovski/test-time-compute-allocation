"""Config-hashed, resumable candidate generation. Orchestrates a Backend
+ PoolStore to fill a pool to N samples per problem, skipping whatever's
already checkpointed -- the same pattern proven out manually in
notes/scratch/day4_generate_pool.py (a killed 100-problem x N=32 run
resumed cleanly there), now as a real, reusable module per Day 6.

Deliberately NOT invoked against any benchmark by this commit -- building
the resumable-sweep mechanism is independent of the pending H1/H4
mentor-input question; actually starting new pool generation is not.
"""

from __future__ import annotations

import concurrent.futures
from dataclasses import dataclass

from marginal_token.backends.base import Backend, DecodeConfig, Sample
from marginal_token.pools.store import Pool, PoolStore, compute_pool_id


@dataclass
class SweepTask:
    problem_id: str
    prompt: str


@dataclass
class SweepResult:
    completed: int
    failed: list[tuple[str, int, str]]  # (problem_id, sample_idx, error message)


def run_sweep(
    tasks: list[SweepTask],
    n: int,
    backend: Backend,
    cfg: DecodeConfig,
    store: PoolStore,
    policy_ref: str,
    backend_ref: str,
    benchmark_id: str,
    max_workers: int = 6,
) -> SweepResult:
    """Fill every task's pool to `n` samples, skipping whatever's already
    checkpointed in `store`. Resumable by construction: call this again
    after a kill and it picks up exactly where it left off, because
    `store.done_sample_indices` is read fresh each time, not cached.
    """
    to_generate: list[tuple[str, str, int]] = []  # (problem_id, prompt, sample_idx)
    for task in tasks:
        pool_id = compute_pool_id(
            policy_ref, backend_ref, benchmark_id, task.problem_id,
            cfg.temperature, cfg.top_p, cfg.max_tokens, cfg.seed, n,
        )
        done = store.done_sample_indices(pool_id, task.problem_id)
        for i in range(n):
            if i not in done:
                to_generate.append((task.problem_id, task.prompt, i))

    failed: list[tuple[str, int, str]] = []
    completed = 0

    def run_one(item: tuple[str, str, int]) -> tuple[str, int, Sample | None, str | None]:
        problem_id, prompt, sample_idx = item
        try:
            [sample] = backend.generate([prompt], cfg)
            sample.problem_id = problem_id
            sample.sample_idx = sample_idx
            return problem_id, sample_idx, sample, None
        except Exception as e:  # noqa: BLE001 -- deliberately broad, this is a generation-failure boundary
            return problem_id, sample_idx, None, str(e)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        for problem_id, sample_idx, sample, error in ex.map(run_one, to_generate):
            if error is not None:
                failed.append((problem_id, sample_idx, error))
                continue
            pool_id = compute_pool_id(
                policy_ref, backend_ref, benchmark_id, problem_id,
                cfg.temperature, cfg.top_p, cfg.max_tokens, cfg.seed, n,
            )
            pool = store.load(pool_id, problem_id, benchmark_id, policy_ref, backend_ref)
            store.append(pool, sample)
            completed += 1

    return SweepResult(completed=completed, failed=failed)
