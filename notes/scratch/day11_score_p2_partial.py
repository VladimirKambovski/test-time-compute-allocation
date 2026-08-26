"""
Score whatever P2 (OlympiadBench-A) problems are ALREADY complete
(N=32) while the rest of P2 keeps generating in the background --
same incremental pattern that worked cleanly for P1 (Day 7/9). PRM
scoring needs only the problem statement, never the gold answer, so
this is fully independent of OlympiadBench's gold-answer format
(`final_answer`/`is_multiple_answer`/`error`) -- that's a scoring-time
concern for LABELING later, not a blocker for PRM scoring now.

Resumable via the same `PRMScoreStore`/`score_pool` mechanism as
`day10_score_full_p1.py` -- safe to re-run after P2 generates more
complete problems, picks up exactly where it left off, never rescoring
anything twice.
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, "src")

from marginal_token.generation.run_sweeps import fetch_olympiad_bench_problems  # noqa: E402
from marginal_token.pools.store import PoolStore  # noqa: E402
from marginal_token.scoring.pipeline import PRMScoreStore, score_pool  # noqa: E402
from marginal_token.scoring.prm_client import HostedQwen25MathPRMClient  # noqa: E402

POOL_ROOT = "results/pools"
SCORE_ROOT = "results/scores"


def main():
    complete_meta = []
    for jsonl_path in sorted(Path(POOL_ROOT).glob("*/*.jsonl")):
        pool_id = jsonl_path.parent.name
        problem_id = jsonl_path.stem
        if not problem_id.isdigit():
            continue  # not an OlympiadBench pool (bare numeric IDs) -- skip MATH-500's
        with open(jsonl_path) as f:
            n_lines = sum(1 for line in f if line.strip())
        if n_lines == 32:
            complete_meta.append((pool_id, problem_id))
    print(f"{len(complete_meta)} P2 problems currently complete (of 300) -- scoring these now")

    if not complete_meta:
        print("nothing complete yet, nothing to do")
        return

    problem_text = dict(fetch_olympiad_bench_problems(
        "Hothan/OlympiadBench", "OE_TO_maths_en_COMP", "configs/benchmarks/data/olympiad-a-ids.json"
    ))
    print(f"fetched {len(problem_text)} problem texts")

    pool_store = PoolStore(POOL_ROOT)
    score_store = PRMScoreStore(SCORE_ROOT)
    client = HostedQwen25MathPRMClient()

    total_new = 0
    t0 = time.time()
    for i, (pool_id, problem_id) in enumerate(complete_meta, 1):
        pool = pool_store.load(pool_id, problem_id, "olympiad-a", "qwen3.5-4b", "policy_primary")
        assert len(pool) == 32
        new = score_pool(pool, query=problem_text[problem_id], client=client, store=score_store)
        total_new += len(new)
        if i % 25 == 0 or i == len(complete_meta):
            elapsed = time.time() - t0
            print(f"  {i}/{len(complete_meta)} problems, {total_new} new scores, {elapsed/60:.1f}min elapsed")

    print(f"done: {total_new} new scores, {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
