"""
Day 10: score the FULL P1 pool (500 problems x 32 = 16,000 samples)
with the primary PRM, via the productionized `scoring/` module (Day 7),
not scratch code -- this is the real infra, exercised at real scale for
the first time. Resumable: `results/scores/` already has ~1,792 samples
from earlier smoke tests (Day 5/7/9), and `score_pool`'s
`done_sample_indices` check picks up exactly where those left off.

Concurrency matches Day 5's validated-safe value for this endpoint
(10) -- the PRM server handled 3,520 calls with zero failures that day.
"""

import concurrent.futures
import json
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, "src")

from marginal_token.pools.store import PoolStore  # noqa: E402
from marginal_token.scoring.pipeline import PRMScoreStore, score_pool  # noqa: E402
from marginal_token.scoring.prm_client import HostedQwen25MathPRMClient  # noqa: E402

POOL_ROOT = "results/pools"
SCORE_ROOT = "results/scores"
MAX_WORKERS = 10


def fetch_all_problem_text(unique_ids):
    wanted = set(unique_ids)
    found = {}
    offset = 0
    while offset < 500 and len(found) < len(wanted):
        url = (f"https://datasets-server.huggingface.co/rows?dataset=HuggingFaceH4/MATH-500"
               f"&config=default&split=test&offset={offset}&length=100")
        d = {}
        for _attempt in range(3):
            try:
                with urllib.request.urlopen(url, timeout=30) as resp:
                    d = json.load(resp)
                break
            except Exception:
                time.sleep(2)
        rows = d.get("rows", [])
        if not rows:
            break
        for r in rows:
            row = r["row"]
            if row["unique_id"] in wanted:
                found[row["unique_id"]] = row["problem"]
        offset += 100
    return found


def main():
    pool_meta = []
    for jsonl_path in sorted(Path(POOL_ROOT).glob("*/*.jsonl")):
        pool_id = jsonl_path.parent.name
        problem_id = jsonl_path.stem.replace("__", "/")
        pool_meta.append((pool_id, problem_id))
    print(f"{len(pool_meta)} pool files found under {POOL_ROOT} (may include P2/OlympiadBench, "
          f"which shares this same root -- filtered out below, not scored here)")

    problem_text = fetch_all_problem_text([pid for _, pid in pool_meta])
    print(f"fetched {len(problem_text)} MATH-500 problem texts")

    # Real bug caught before it could crash the run (2026-08-23): P2's
    # OlympiadBench generation writes into this SAME `results/pools/`
    # root concurrently, and this glob doesn't distinguish benchmarks --
    # it picked up P2's in-progress "1606" pool too, which would have
    # KeyError'd on `problem_text["1606"]` (not a MATH-500 ID) partway
    # through the run. Filter to exactly the pools this script actually
    # fetched text for, rather than trusting the glob's benchmark scope.
    before = len(pool_meta)
    pool_meta = [(pool_id, pid) for pool_id, pid in pool_meta if pid in problem_text]
    if len(pool_meta) != before:
        print(f"filtered out {before - len(pool_meta)} non-MATH-500 pool file(s) found under the shared root")
    assert len(pool_meta) == 500, f"expected exactly 500 real P1 pools after filtering, got {len(pool_meta)}"

    pool_store = PoolStore(POOL_ROOT)
    score_store = PRMScoreStore(SCORE_ROOT)

    def score_one_problem(item):
        pool_id, problem_id = item
        client = HostedQwen25MathPRMClient()  # one client per thread -- cheap, avoids any shared-state risk
        pool = pool_store.load(pool_id, problem_id, "math500", "qwen3.5-4b", "policy_primary")
        new_scores = score_pool(pool, query=problem_text[problem_id], client=client, store=score_store)
        return problem_id, len(new_scores)

    t0 = time.time()
    total_new = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = [ex.submit(score_one_problem, item) for item in pool_meta]
        n_done_problems = 0
        for fut in concurrent.futures.as_completed(futures):
            problem_id, n_new = fut.result()
            total_new += n_new
            n_done_problems += 1
            if n_done_problems % 25 == 0 or n_done_problems == len(pool_meta):
                elapsed = time.time() - t0
                print(f"  {n_done_problems}/{len(pool_meta)} problems processed, "
                      f"{total_new} new scores so far, {elapsed/60:.1f}min elapsed")

    print(f"done: {total_new} new scores written, {time.time()-t0:.1f}s total")


if __name__ == "__main__":
    main()
