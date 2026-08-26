"""
Day 13, digging into the max_tokens angle: select ~12 MATH-500 (P1)
problems from difficulty band 0 (the hardest 20%, where the E2 landscape
found a 0.000 ceiling for every action including the oracle, and where
today's survivorship check found only 0.21/32 mean successful
extractions -- essentially total truncation-driven failure). Tests
whether that "unwinnable" ceiling is a token-budget artifact rather than
a genuine reasoning-difficulty ceiling: does raising max_tokens actually
change the oracle action distribution (fewer ABSTAIN, more real
STOP/SAMPLE/SELECT), not just the raw truncation rate (already checked
at the aggregate level on Day 3).

Same streaming, memory-safe pattern as prior Day 12/13 scripts.
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
    if len(pool) != 32:
        return None
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
        p1 = pass_at_1_one_pool(store, pool_id, pid, "math500", math_gold[pid])
        if p1 is None:
            continue
        pass_at_1[f"math500:{pid}"] = p1
        n_processed += 1
        if n_processed % 100 == 0:
            gc.collect()

    for pool_id, pid in oly_meta:
        if pid not in oly_gold:
            continue
        p1 = pass_at_1_one_pool(store, pool_id, pid, "olympiad-a", oly_gold[pid])
        if p1 is None:
            continue
        pass_at_1[f"olympiad-a:{pid}"] = p1
        n_processed += 1
        if n_processed % 100 == 0:
            gc.collect()

    print(f"usable: {n_processed} total (across both benchmarks)")
    bands = difficulty_bands(pass_at_1, n_bands=5)

    band0_math = sorted(
        [(k.split(":", 1)[1], pass_at_1[k]) for k, b in bands.items() if b == 0 and k.startswith("math500:")],
        key=lambda kv: (kv[1], kv[0]),
    )
    print(f"band 0 (hardest) total problems: {sum(1 for b in bands.values() if b == 0)}")
    print(f"band 0 MATH-500-only problems: {len(band0_math)}")

    TARGET_N = 12
    stride = max(1, len(band0_math) // TARGET_N)
    selected = band0_math[::stride][:TARGET_N]

    print(f"\nselected {len(selected)} problems for max_tokens ablation (band 0, MATH-500):")
    for pid, p1 in selected:
        print(f"  {pid}  pass@1={p1:.3f}")

    ids_file = "notes/scratch/day13_maxtokens_ablation_ids.json"
    json.dump([pid for pid, _ in selected], open(ids_file, "w"), indent=2)
    print(f"\nwrote {ids_file}")


if __name__ == "__main__":
    main()
