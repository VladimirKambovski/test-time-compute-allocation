"""
Offline batch PRM scoring -- reads an already-complete `Pool`
(`pools/store.py`), segments + scores every sample against a `PRMClient`,
and persists per-sample results (including the full per-step reward
array, not just the mean -- the roadmap's explicit Day-7 ask) to a
separate, resumable store.

Deliberately a separate process/module from generation (docs/brief.md
§22's module boundary): this reads already-generated, already-persisted
samples. It never triggers generation itself, and can run at any time
after a pool (or even just part of one) exists -- exactly how this was
smoke-tested, against a handful of P1 problems that had already
completed while the rest of P1 kept generating in the background
(`notes/2026-08-23.md`).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from marginal_token.answers.taxonomy import FailureStatus
from marginal_token.backends.base import Sample
from marginal_token.pools.store import Pool
from marginal_token.scoring.prm_client import PRMClient
from marginal_token.scoring.segmentation import DEFAULT_CONVENTION, segment


def compute_score_id(pool_id: str, prm_role: str, convention: str) -> str:
    """Composite identity for a scoring run: which pool, which PRM
    (ladder rung/role), and which segmentation convention. Mirrors
    `pools.store.compute_pool_id`'s discipline (invariant #3) -- a change
    to any of these three fields is a NEW set of scores, never silently
    merged with (or mistaken as already covering) a different
    combination's work. Without this, re-scoring the same pool under a
    different PRM rung or segmentation convention would collide with
    the first run's checkpoint files and `PRMScoreStore.done_sample_indices`
    would wrongly report those samples as already scored.
    """
    parts = [pool_id, prm_role, convention]
    return hashlib.blake2s("\x1f".join(parts).encode()).hexdigest()


@dataclass
class PRMScore:
    problem_id: str
    sample_idx: int
    convention: str
    prm_role: str  # which ladder rung/role scored this, e.g. "primary_prm"
    status: str  # FailureStatus.OK / STEP_SEGMENTATION_FAILED / PRM_SCORE_MISSING
    num_steps: int | None = None
    step_rewards: list[float] | None = None  # per-step array -- Day 7's explicit ask
    mean_reward: float | None = None
    final_step_reward: float | None = None
    error: str | None = None

    def as_dict(self) -> dict:
        return {
            "problem_id": self.problem_id,
            "sample_idx": self.sample_idx,
            "convention": self.convention,
            "prm_role": self.prm_role,
            "status": self.status,
            "num_steps": self.num_steps,
            "step_rewards": self.step_rewards,
            "mean_reward": self.mean_reward,
            "final_step_reward": self.final_step_reward,
            "error": self.error,
        }

    @staticmethod
    def from_dict(d: dict) -> PRMScore:
        return PRMScore(**d)


def score_sample(
    sample: Sample, query: str, client: PRMClient, convention: str = DEFAULT_CONVENTION
) -> PRMScore:
    """Score exactly one sample. Never raises on a segmentation or PRM
    failure -- maps to the closed taxonomy instead (invariant #7), never
    a silent 0/incorrect score.
    """
    prm_role = getattr(client, "role", "unknown")
    steps = segment(sample.text, convention)
    if not steps:
        return PRMScore(
            problem_id=sample.problem_id, sample_idx=sample.sample_idx, convention=convention,
            prm_role=prm_role, status=FailureStatus.STEP_SEGMENTATION_FAILED.value,
        )
    result = client.score(query, steps)
    if not result.ok:
        return PRMScore(
            problem_id=sample.problem_id, sample_idx=sample.sample_idx, convention=convention,
            prm_role=prm_role, status=FailureStatus.PRM_SCORE_MISSING.value, error=result.error,
        )
    return PRMScore(
        problem_id=sample.problem_id, sample_idx=sample.sample_idx, convention=convention,
        prm_role=prm_role, status=FailureStatus.OK.value, num_steps=result.num_steps,
        step_rewards=result.step_rewards, mean_reward=result.mean_reward,
        final_step_reward=result.step_rewards[-1] if result.step_rewards else None,
    )


class PRMScoreStore:
    """Filesystem-backed score store, content-addressed by
    `compute_score_id` (pool + PRM role + convention). One JSONL file per
    (score_id, problem_id), append-only -- same checkpoint-per-write,
    resumable-by-construction pattern as `pools.store.PoolStore`.
    """

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, score_id: str, problem_id: str) -> Path:
        safe_problem_id = problem_id.replace("/", "__")
        return self.root / score_id / f"{safe_problem_id}.jsonl"

    def done_sample_indices(self, score_id: str, problem_id: str) -> set[int]:
        path = self._path(score_id, problem_id)
        if not path.exists():
            return set()
        indices = set()
        with open(path) as f:
            for line in f:
                if line.strip():
                    indices.add(json.loads(line)["sample_idx"])
        return indices

    def append(self, score_id: str, problem_id: str, score: PRMScore) -> None:
        path = self._path(score_id, problem_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a") as f:
            f.write(json.dumps(score.as_dict()) + "\n")

    def load(self, score_id: str, problem_id: str) -> list[PRMScore]:
        path = self._path(score_id, problem_id)
        if not path.exists():
            return []
        with open(path) as f:
            return [PRMScore.from_dict(json.loads(line)) for line in f if line.strip()]


def score_pool(
    pool: Pool,
    query: str,
    client: PRMClient,
    store: PRMScoreStore,
    convention: str = DEFAULT_CONVENTION,
) -> list[PRMScore]:
    """Score every sample in `pool` not already scored under this exact
    (pool, PRM role, convention) combination. Resumable: `done_sample_indices`
    is re-read fresh each call, same pattern as `generation.sweep.run_sweep`
    -- calling this again after a kill picks up exactly where it left off.
    Returns only the scores generated THIS call; use `store.load(...)` for
    the full accumulated set.
    """
    prm_role = getattr(client, "role", "unknown")
    score_id = compute_score_id(pool.pool_id, prm_role, convention)
    done = store.done_sample_indices(score_id, pool.problem_id)
    new_scores = []
    for sample in pool.samples:
        if sample.sample_idx in done:
            continue
        score = score_sample(sample, query, client, convention)
        store.append(score_id, pool.problem_id, score)
        new_scores.append(score)
    return new_scores
