"""
Day 18: PRM-score the held-out pools (P4: OlympiadBench-B, 100
problems; P5: AIME25, 30 problems). Real infra (scoring/pipeline.py),
same pattern as day10_score_full_p1.py, but pool enumeration is built
EXPLICITLY from the frozen id lists via compute_pool_id -- never a
glob over results/pools/*/*.jsonl. Two independent reasons this
matters here specifically, not just general caution:
  1. day10_score_full_p1.py's own docstring documents a real bug where
     a glob picked up a DIFFERENT benchmark's in-progress pool from the
     same shared root.
  2. Tonight's duplicate-pool bug (notes/2026-08-26.md) showed a glob
     can silently double-count problems that have stray non-canonical
     directories on disk.
Both risks are structurally impossible here since pool_meta is built
from the known 100+30 canonical ids directly, not discovered.
"""
import concurrent.futures
import json
import sys
import time

sys.path.insert(0, "src")

from marginal_token.pools.store import PoolStore, compute_pool_id  # noqa: E402
from marginal_token.scoring.pipeline import PRMScoreStore, score_pool  # noqa: E402
from marginal_token.scoring.prm_client import HostedQwen25MathPRMClient  # noqa: E402
from marginal_token.generation.run_sweeps import fetch_olympiad_bench_problems, fetch_aime25_problems  # noqa: E402

POOL_ROOT = "results/pools"
SCORE_ROOT = "results/scores"
MAX_WORKERS = 10  # Day 5/10's validated-safe PRM concurrency, zero failures at this value historically


def main():
    print("fetching P4 (OlympiadBench-B) + P5 (AIME25) problem texts...", flush=True)
    p4_rows = fetch_olympiad_bench_problems(
        "Hothan/OlympiadBench", "OE_TO_maths_en_COMP", "configs/benchmarks/data/heldout-olympiad-b-ids.json",
    )
    p5_rows = fetch_aime25_problems(
        "a6ad95f611d72cf628a80b58bd0432ef6638f958", "configs/benchmarks/data/heldout-aime25-ids.json",
    )
    print(f"P4: {len(p4_rows)}/100, P5: {len(p5_rows)}/30", flush=True)
    assert len(p4_rows) == 100, f"expected exactly 100 P4 problems, got {len(p4_rows)}"
    assert len(p5_rows) == 30, f"expected exactly 30 P5 problems, got {len(p5_rows)}"

    items = []  # (pool_id, problem_id, benchmark_id, query_text)
    for pid, text in p4_rows:
        pool_id = compute_pool_id(policy_ref="qwen3.5-4b", backend_ref="policy_primary", benchmark_id="olympiad-b-heldout",
                                    problem_id=pid, temperature=0.8, top_p=0.95, max_tokens=1024, seed=None, n=32)
        items.append((pool_id, pid, "olympiad-b-heldout", text))
    for pid, text in p5_rows:
        pool_id = compute_pool_id(policy_ref="qwen3.5-4b", backend_ref="policy_primary", benchmark_id="aime25-heldout",
                                    problem_id=pid, temperature=0.8, top_p=0.95, max_tokens=1024, seed=None, n=32)
        items.append((pool_id, pid, "aime25-heldout", text))

    pool_store = PoolStore(POOL_ROOT)
    score_store = PRMScoreStore(SCORE_ROOT)

    def score_one(item):
        pool_id, problem_id, benchmark_id, text = item
        client = HostedQwen25MathPRMClient()
        pool = pool_store.load(pool_id, problem_id, benchmark_id, "qwen3.5-4b", "policy_primary")
        assert len(pool) == 32, f"{problem_id}: expected 32 samples, got {len(pool)}"
        new_scores = score_pool(pool, query=text, client=client, store=score_store)
        return problem_id, len(new_scores)

    t0 = time.time()
    total_new = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = [ex.submit(score_one, item) for item in items]
        n_done = 0
        for fut in concurrent.futures.as_completed(futures):
            problem_id, n_new = fut.result()
            total_new += n_new
            n_done += 1
            if n_done % 25 == 0 or n_done == len(items):
                elapsed = time.time() - t0
                print(f"  {n_done}/{len(items)} problems scored, {total_new} new scores, {elapsed/60:.1f}min", flush=True)

    print(f"\ndone: {total_new} new scores written, {time.time()-t0:.1f}s total")


if __name__ == "__main__":
    main()
