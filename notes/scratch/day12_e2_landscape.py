"""
Day 12: E2 action-value landscape across {STOP,SAMPLE,SELECT} x 5
difficulty bands, on P1 union P2 at N=32. Pure local analysis -- both
pools are fully generated and fully PRM-scored, zero new network calls
beyond gold-answer fetches.

**Rewritten (2026-08-24) to be memory-safe.** The first version loaded
every pool for both benchmarks into in-memory lists before processing --
harmless for P1 (no logprobs), but P2's samples carry real per-token
logprob arrays (the Day 10 backend fix), and holding all ~800 pools'
worth of that in memory at once used 9.4GB+ RAM and filled swap on a
13GB machine, found live when it started visibly starving the system.
Fixed by processing ONE pool at a time in a single streaming loop --
only small scalar results (a label, a pass@1 float) are retained across
problems, never the full `Pool`/`Sample` objects themselves.
"""

import gc
import json
import sys
import time
import urllib.request
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")

from marginal_token.answers.equivalence import check_equivalent  # noqa: E402
from marginal_token.answers.extraction import extract_answer  # noqa: E402
from marginal_token.answers.taxonomy import FailureStatus  # noqa: E402
from marginal_token.controller.oracle_labels import oracle_action_label  # noqa: E402
from marginal_token.evaluation.stats import difficulty_bands  # noqa: E402
from marginal_token.generation.run_sweeps import fetch_olympiad_bench_problems  # noqa: E402
from marginal_token.pools.store import PoolStore  # noqa: E402

POOL_ROOT = "results/pools"


def fetch_math500_gold(unique_ids):
    wanted = set(unique_ids)
    found = {}
    offset = 0
    while offset < 500 and len(found) < len(wanted):
        url = (f"https://datasets-server.huggingface.co/rows?dataset=HuggingFaceH4/MATH-500"
               f"&config=default&split=test&offset={offset}&length=100")
        d = {}
        for _ in range(3):
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
                found[row["unique_id"]] = row["answer"]
        offset += 100
    return found


def fetch_olympiad_gold(ids):
    wanted = set(ids)
    found = {}
    offset = 0
    while offset < 674 and len(found) < len(wanted):
        url = (f"https://datasets-server.huggingface.co/rows?dataset=Hothan/OlympiadBench"
               f"&config=OE_TO_maths_en_COMP&split=train&offset={offset}&length=100")
        d = {}
        for _ in range(3):
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
            row_id = str(row["id"])
            if row_id in wanted and not row.get("is_multiple_answer") and not row.get("error") and row.get("final_answer"):
                found[row_id] = row["final_answer"][0]
        offset += 100
    return found


def process_one_pool(store, pool_id, pid, benchmark_id, gold):
    """Load exactly ONE pool, compute everything needed from it, then let
    it go out of scope -- never held alongside any other pool's samples.
    """
    pool = store.load(pool_id, pid, benchmark_id, "qwen3.5-4b", "policy_primary")
    assert len(pool) == 32

    c = 0
    for s in pool.samples:
        ext = extract_answer(s.text, finish_reason=s.finish_reason)
        if ext.status == FailureStatus.OK and check_equivalent(prediction=ext.value, gold=gold).equivalent:
            c += 1
    pass_at_1 = c / 32

    label_result = oracle_action_label(pool.samples, gold)
    return pass_at_1, label_result


def main():
    math_meta, oly_meta = [], []
    for jsonl_path in sorted(Path(POOL_ROOT).glob("*/*.jsonl")):
        pool_id = jsonl_path.parent.name
        problem_id = jsonl_path.stem.replace("__", "/")
        (oly_meta if problem_id.isdigit() else math_meta).append((pool_id, problem_id))
    print(f"P1 (MATH-500): {len(math_meta)} problems, P2 (OlympiadBench-A): {len(oly_meta)} problems")

    math_gold = fetch_math500_gold([pid for _, pid in math_meta])
    oly_gold = fetch_olympiad_gold([pid for _, pid in oly_meta])
    print(f"gold answers: {len(math_gold)} MATH-500, {len(oly_gold)}/{len(oly_meta)} OlympiadBench-A usable")

    store = PoolStore(POOL_ROOT)

    pass_at_1 = {}
    label_results = {}  # key -> OracleLabelResult
    n_processed = 0
    total = sum(1 for _, pid in math_meta if pid in math_gold) + sum(1 for _, pid in oly_meta if pid in oly_gold)

    for pool_id, pid in math_meta:
        if pid not in math_gold:
            continue
        key = f"math500:{pid}"
        p1, label_result = process_one_pool(store, pool_id, pid, "math500", math_gold[pid])
        pass_at_1[key] = p1
        label_results[key] = label_result
        n_processed += 1
        if n_processed % 100 == 0:
            gc.collect()
            print(f"  {n_processed}/{total} problems processed")

    for pool_id, pid in oly_meta:
        if pid not in oly_gold:
            continue
        key = f"olympiad-a:{pid}"
        p1, label_result = process_one_pool(store, pool_id, pid, "olympiad-a", oly_gold[pid])
        pass_at_1[key] = p1
        label_results[key] = label_result
        n_processed += 1
        if n_processed % 100 == 0:
            gc.collect()
            print(f"  {n_processed}/{total} problems processed")

    print(f"\nusable: {n_processed} total (across both benchmarks)")

    bands = difficulty_bands(pass_at_1, n_bands=5)

    landscape = defaultdict(lambda: defaultdict(list))
    oracle_dist = defaultdict(int)
    for key, result in label_results.items():
        band = bands[key]
        oracle_dist[result.action] += 1
        landscape[band]["stop"].append(result.stop_correct)
        landscape[band]["sample"].append(result.sample_correct)
        landscape[band]["select"].append(result.select_correct)

    print(f"\noracle action distribution (P1 union P2, N=32): {dict(oracle_dist)}")
    total_n = sum(oracle_dist.values())
    print(f"SELECT-only rate: {oracle_dist.get('select', 0)}/{total_n} = {oracle_dist.get('select', 0)/total_n:.1%}")

    print(f"\n=== E2 landscape: accuracy by difficulty band (0=hardest, 4=easiest) ===")
    print(f"{'band':>6}{'n':>6}{'STOP':>8}{'SAMPLE':>8}{'SELECT_ceil':>13}{'oracle':>8}{'best_fixed':>12}{'gap(pp)':>10}")
    for band in range(5):
        stop_b = np.array(landscape[band]["stop"])
        sample_b = np.array(landscape[band]["sample"])
        select_b = np.array(landscape[band]["select"])
        n = len(stop_b)
        oracle_b = (stop_b | sample_b | select_b).mean()
        best_fixed_b = max(stop_b.mean(), sample_b.mean())
        gap_b = 100 * (oracle_b - best_fixed_b)
        print(f"{band:>6}{n:>6}{stop_b.mean():>8.3f}{sample_b.mean():>8.3f}{select_b.mean():>13.3f}"
              f"{oracle_b:>8.3f}{best_fixed_b:>12.3f}{gap_b:>10.2f}")

    print("\n(Full P1+P2 landscape at N=32 fixed budget. The 5-budget-level axis from E2's full "
          "design is a natural follow-up; this pass establishes the difficulty-band structure and "
          "the combined-benchmark oracle distribution.)")


if __name__ == "__main__":
    main()
