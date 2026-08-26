"""
Day 13: select ~20-25 MATH-500 (P1) problems from difficulty band 1
(the moderately-hard band Day 12's E2 landscape found carries ~4pp of
the G1 gap) for the temperature ablation (temp=1.0 vs. frozen 0.8,
paired against existing P1 data for the same problems).

Reuses day12_e2_landscape.py's exact band-computation logic (same
streaming, memory-safe, one-pool-at-a-time pattern) so band assignment
is identical to the one already reported in HANDOFF.md/notes -- not a
re-derivation that could silently drift from it.
"""

import gc
import json
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, "src")

from marginal_token.answers.equivalence import check_equivalent  # noqa: E402
from marginal_token.answers.extraction import extract_answer  # noqa: E402
from marginal_token.answers.taxonomy import FailureStatus  # noqa: E402
from marginal_token.evaluation.stats import difficulty_bands  # noqa: E402
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


def pass_at_1_one_pool(store, pool_id, pid, benchmark_id, gold):
    pool = store.load(pool_id, pid, benchmark_id, "qwen3.5-4b", "policy_primary")
    assert len(pool) == 32
    c = 0
    for s in pool.samples:
        ext = extract_answer(s.text, finish_reason=s.finish_reason)
        if ext.status == FailureStatus.OK and check_equivalent(prediction=ext.value, gold=gold).equivalent:
            c += 1
    return c / 32


def main():
    math_meta, oly_meta = [], []
    for jsonl_path in sorted(Path(POOL_ROOT).glob("*/*.jsonl")):
        pool_id = jsonl_path.parent.name
        problem_id = jsonl_path.stem.replace("__", "/")
        (oly_meta if problem_id.isdigit() else math_meta).append((pool_id, problem_id))

    math_gold = fetch_math500_gold([pid for _, pid in math_meta])
    oly_gold = fetch_olympiad_gold([pid for _, pid in oly_meta])

    store = PoolStore(POOL_ROOT)
    pass_at_1 = {}
    n_processed = 0

    for pool_id, pid in math_meta:
        if pid not in math_gold:
            continue
        key = f"math500:{pid}"
        pass_at_1[key] = pass_at_1_one_pool(store, pool_id, pid, "math500", math_gold[pid])
        n_processed += 1
        if n_processed % 100 == 0:
            gc.collect()

    for pool_id, pid in oly_meta:
        if pid not in oly_gold:
            continue
        key = f"olympiad-a:{pid}"
        pass_at_1[key] = pass_at_1_one_pool(store, pool_id, pid, "olympiad-a", oly_gold[pid])
        n_processed += 1
        if n_processed % 100 == 0:
            gc.collect()

    print(f"usable: {n_processed} total (across both benchmarks)")

    bands = difficulty_bands(pass_at_1, n_bands=5)

    band1_math = sorted(
        [(k.split(":", 1)[1], pass_at_1[k]) for k, b in bands.items() if b == 1 and k.startswith("math500:")],
        key=lambda kv: (kv[1], kv[0]),
    )
    print(f"band 1 (moderately-hard) total problems: {sum(1 for b in bands.values() if b == 1)}")
    print(f"band 1 MATH-500-only problems: {len(band1_math)}")

    # Evenly-strided sample across band 1's own pass@1 range (band1_math is
    # already sorted by pass@1) so the ablation covers the band's spread,
    # not just one edge of it.
    TARGET_N = 24
    stride = max(1, len(band1_math) // TARGET_N)
    selected = band1_math[::stride][:TARGET_N]

    print(f"\nselected {len(selected)} problems for temperature ablation (band 1, MATH-500):")
    for pid, p1 in selected:
        print(f"  {pid}  pass@1={p1:.3f}")

    ids_file = "notes/scratch/day13_temp_ablation_ids.json"
    with open(ids_file, "w") as f:
        json.dump([pid for pid, _ in selected], f, indent=2)
    print(f"\nwrote {ids_file}")


if __name__ == "__main__":
    main()
