"""Content-addressed pool store with nested-prefix views.

pool_id = blake2s(policy_ref, backend_ref, benchmark_id, problem_id,
                   temp, top_p, max_tokens, seed, N) per docs/brief.md §13.

N=32 is the current MUST floor (changed from N=64 on 2026-08-20 by
explicit user instruction -- see notes/2026-08-20.md); N=64 is now the
SHOULD extension. Nested prefixes: a pool of N samples supports
truncation to any k<=N with zero extra generation (invariant #2) --
`Pool.prefix(k)` is that read, not a new generation.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

from marginal_token.backends.base import Provenance, Sample

DEFAULT_MUST_FLOOR_N = 32  # see CLAUDE.md invariant #1 (updated 2026-08-20)


def compute_pool_id(
    policy_ref: str,
    backend_ref: str,
    benchmark_id: str,
    problem_id: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    seed: int | None,
    n: int,
) -> str:
    """blake2s over the exact ordered fields in docs/brief.md §13. Any
    change to any of these fields produces a different pool_id -- by
    design, since a change to any of them means a NEW pool, never a
    silent continuation (invariant #3).
    """
    parts = [
        policy_ref, backend_ref, benchmark_id, problem_id,
        repr(temperature), repr(top_p), str(max_tokens), repr(seed), str(n),
    ]
    h = hashlib.blake2s("\x1f".join(parts).encode())
    return h.hexdigest()


class PoolContractError(ValueError):
    """Raised when a sample's contract fields disagree with the pool's
    manifest -- never silently dropped or merged. See §27.6.
    """


@dataclass
class PoolManifest:
    """The frozen condition every sample in a pool must match, per
    §27.6's compatibility contract.
    """

    policy_ref: str
    backend_ref: str
    benchmark_id: str
    contract_key: tuple  # Provenance.contract_key() of the first sample admitted


@dataclass
class Pool:
    """One problem's frozen sample pool. Samples are stored in
    generation order (sample_idx 0..N-1) -- `prefix(k)` reads the first k
    without any extra generation, which is the entire point of nested
    prefixes.
    """

    problem_id: str
    pool_id: str
    manifest: PoolManifest
    samples: list[Sample] = field(default_factory=list)

    def add(self, sample: Sample) -> None:
        """Validate against the manifest before admitting -- reject
        (raise), don't silently drop, a contract-mismatched sample.
        """
        key = sample.provenance.contract_key()
        if self.manifest.contract_key is None:
            self.manifest.contract_key = key
        elif key != self.manifest.contract_key:
            raise PoolContractError(
                f"Sample for {self.problem_id} sample_idx={sample.sample_idx} has contract "
                f"fields {key} which disagree with this pool's manifest "
                f"{self.manifest.contract_key}. A backend/model/decode-config change is a NEW "
                f"pool, never a silent continuation of this one (invariant #3, §27.6)."
            )
        self.samples.append(sample)

    def prefix(self, k: int) -> list[Sample]:
        """Nested-prefix read: any k<=N from an already-generated pool,
        zero extra generation (invariant #2).
        """
        if k > len(self.samples):
            raise ValueError(
                f"Requested prefix k={k} but pool {self.pool_id} only has {len(self.samples)} "
                f"samples. Nested prefixes only support k<=N -- this is not a request to generate "
                f"more, that would be a new pool extension, done explicitly."
            )
        return self.samples[:k]

    def __len__(self) -> int:
        return len(self.samples)


class PoolStore:
    """Filesystem-backed content-addressed store. One JSONL file per
    (problem, pool_id) -- append-only, matching the checkpointed-generation
    pattern already proven out in notes/scratch/day4_generate_pool.py.
    """

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, pool_id: str, problem_id: str) -> Path:
        safe_problem_id = problem_id.replace("/", "__")
        return self.root / pool_id / f"{safe_problem_id}.jsonl"

    def load(self, pool_id: str, problem_id: str, benchmark_id: str, policy_ref: str, backend_ref: str) -> Pool:
        path = self._path(pool_id, problem_id)
        manifest = PoolManifest(policy_ref=policy_ref, backend_ref=backend_ref,
                                 benchmark_id=benchmark_id, contract_key=None)
        pool = Pool(problem_id=problem_id, pool_id=pool_id, manifest=manifest)
        if path.exists():
            with open(path) as f:
                for line in f:
                    if line.strip():
                        pool.add(_sample_from_dict(json.loads(line)))
        return pool

    def append(self, pool: Pool, sample: Sample) -> None:
        """Validate + append to the in-memory Pool AND persist immediately
        -- matches the checkpoint-per-write pattern that made the Day 4
        generation runs resumable after a kill/timeout.
        """
        pool.add(sample)  # raises on contract mismatch before we ever touch disk
        path = self._path(pool.pool_id, pool.problem_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a") as f:
            f.write(json.dumps(_sample_to_dict(sample)) + "\n")

    def done_sample_indices(self, pool_id: str, problem_id: str) -> set[int]:
        """For resumability: which sample_idx values already exist,
        without needing to reconstruct full Sample objects.
        """
        path = self._path(pool_id, problem_id)
        if not path.exists():
            return set()
        indices = set()
        with open(path) as f:
            for line in f:
                if line.strip():
                    indices.add(json.loads(line)["sample_idx"])
        return indices


def _sample_to_dict(sample: Sample) -> dict:
    return {
        "text": sample.text,
        "finish_reason": sample.finish_reason,
        "completion_tokens": sample.completion_tokens,
        "prompt_tokens": sample.prompt_tokens,
        "provenance": sample.provenance.as_dict(),
        "logprobs": sample.logprobs,
        "problem_id": sample.problem_id,
        "sample_idx": sample.sample_idx,
        "extra": sample.extra,
    }


def _sample_from_dict(d: dict) -> Sample:
    return Sample(
        text=d["text"],
        finish_reason=d["finish_reason"],
        completion_tokens=d["completion_tokens"],
        prompt_tokens=d["prompt_tokens"],
        provenance=Provenance(**d["provenance"]),
        logprobs=d.get("logprobs"),
        problem_id=d.get("problem_id", ""),
        sample_idx=d.get("sample_idx", 0),
        extra=d.get("extra", {}),
    )
