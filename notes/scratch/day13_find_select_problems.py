"""
Day 13, prompted by the mentor's actual suggestion (relayed 2026-08-26):
raise temperature, use a smaller pool, and target problems where SELECT
is EXPECTED to win -- not just "the hardest band" (that was this
session's own proxy, a different criterion). Find the real P1 problems
whose true 4-class oracle label is 'select' at the frozen temp=0.8, plus
their extraction health (so we don't walk into the same
truncation-survivorship confound found earlier today).
"""
import gc
import json
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, "src")

from marginal_token.answers.extraction import extract_answer  # noqa: E402
from marginal_token.answers.taxonomy import FailureStatus  # noqa: E402
from marginal_token.controller.oracle_labels import oracle_action_label  # noqa: E402
from marginal_token.pools.store import PoolStore  # noqa: E402

POOL_ROOT = "results/pools"


def fetch_math500_gold_all():
    found = {}
    offset = 0
    while offset < 500:
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
            found[row["unique_id"]] = row["answer"]
        offset += 100
    return found


def main():
    math_meta = []
    for jsonl_path in sorted(Path(POOL_ROOT).glob("*/*.jsonl")):
        pool_id = jsonl_path.parent.name
        problem_id = jsonl_path.stem.replace("__", "/")
        if not problem_id.isdigit():
            math_meta.append((pool_id, problem_id))

    gold = fetch_math500_gold_all()
    store = PoolStore(POOL_ROOT)

    select_problems = []
    n_processed = 0
    for pool_id, pid in math_meta:
        if pid not in gold:
            continue
        pool = store.load(pool_id, pid, "math500", "qwen3.5-4b", "policy_primary")
        if len(pool) != 32:
            continue
        result = oracle_action_label(pool.samples, gold[pid])
        if result.action == "select":
            n_ok = sum(1 for s in pool.samples if extract_answer(s.text, finish_reason=s.finish_reason).status == FailureStatus.OK)
            select_problems.append((pid, n_ok))
        n_processed += 1
        if n_processed % 100 == 0:
            gc.collect()
            print(f"  {n_processed} processed")

    print(f"\ntotal P1 problems processed: {n_processed}")
    print(f"true SELECT-oracle problems in P1 (temp=0.8): {len(select_problems)}")
    for pid, n_ok in select_problems:
        print(f"  {pid}: successful extractions = {n_ok}/32")

    ids_file = "notes/scratch/day13_select_oracle_ids.json"
    json.dump([pid for pid, _ in select_problems], open(ids_file, "w"), indent=2)
    print(f"\nwrote {ids_file}")


if __name__ == "__main__":
    main()
