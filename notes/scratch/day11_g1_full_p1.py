"""
Gate G1 itself (oracle-over-{STOP,SAMPLE,SELECT} minus best fixed
policy), rerun at the FULL 500-problem P1 scale -- the original G1
decision (notes/2026-08-21.md) was made on the 100-problem dev-100
subset. Pure local analysis, reusing Day 10's `oracle_action_label()`
(now the real 4-class version) rather than re-deriving the STOP/SAMPLE/
SELECT logic a third time.
"""

import json
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")

from marginal_token.controller.oracle_labels import oracle_action_label  # noqa: E402
from marginal_token.evaluation.stats import paired_bootstrap_bca  # noqa: E402
from marginal_token.pools.store import PoolStore  # noqa: E402

POOL_ROOT = "results/pools"


def fetch_all_gold(unique_ids):
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
                found[row["unique_id"]] = row["answer"]
        offset += 100
    return found


def main():
    pool_meta = []
    for jsonl_path in sorted(Path(POOL_ROOT).glob("*/*.jsonl")):
        pool_id = jsonl_path.parent.name
        problem_id = jsonl_path.stem.replace("__", "/")
        if problem_id.isdigit():
            continue
        pool_meta.append((pool_id, problem_id))
    assert len(pool_meta) == 500

    gold = fetch_all_gold([pid for _, pid in pool_meta])
    print(f"fetched {len(gold)} gold answers")

    store = PoolStore(POOL_ROOT)
    stop_arr, sample_arr, select_arr = [], [], []
    label_counts = {"stop": 0, "sample": 0, "select": 0, "abstain": 0}

    for pool_id, problem_id in pool_meta:
        pool = store.load(pool_id, problem_id, "math500", "qwen3.5-4b", "policy_primary")
        assert len(pool) == 32
        result = oracle_action_label(pool.samples, gold[problem_id])
        stop_arr.append(result.stop_correct)
        sample_arr.append(result.sample_correct)
        select_arr.append(result.select_correct)
        label_counts[result.action] += 1

    stop_arr = np.array(stop_arr)
    sample_arr = np.array(sample_arr)
    select_arr = np.array(select_arr)  # this is the pass@32 ceiling, same role as G1's "SELECT" oracle proxy
    oracle_correct = stop_arr | sample_arr | select_arr
    best_fixed_acc = max(stop_arr.mean(), sample_arr.mean())
    oracle_acc = oracle_correct.mean()
    gap = (oracle_acc - best_fixed_acc) * 100

    diff = oracle_correct.astype(float) - np.maximum(stop_arr, sample_arr).astype(float)
    boot = paired_bootstrap_bca(diff, n_resamples=10_000, seed=20260824)

    print(f"\n=== Gate G1 at full P1 scale (500 MATH-500 problems, N=32) ===")
    print(f"STOP accuracy:              {stop_arr.mean():.4f}")
    print(f"SAMPLE accuracy:            {sample_arr.mean():.4f}")
    print(f"SELECT ceiling (pass@32):   {select_arr.mean():.4f}")
    print(f"Oracle accuracy:            {oracle_acc:.4f}")
    print(f"Best fixed policy accuracy: {best_fixed_acc:.4f}")
    print(f"GAP: {gap:.2f}pp, 95% bootstrap CI: [{boot.ci_lo*100:.2f}, {boot.ci_hi*100:.2f}]")
    print(f"G1 accept (>=8pp, CI excludes 0): "
          f"{'PASS' if gap >= 8 and boot.ci_lo > 0 else 'still does not clear the threshold'}")

    print(f"\noracle action label distribution (full 500): {label_counts}")
    select_win_rate = label_counts["select"] / 500
    print(f"SELECT-only oracle win rate: {select_win_rate:.1%} "
          f"(cross-check against dev-100's 1% / weaker-policy's 3%, notes/2026-08-21.md/2026-08-22.md)")

    print("\n(Cross-check against the dev-100 subset G1 result, notes/2026-08-21.md: "
          "STOP=0.670, SAMPLE=0.730, SELECT ceiling=0.740, gap=1.00pp, CI [0.00, 3.00]. "
          "This is the FULL 500-problem P1 pool -- a materially larger, non-identical sample.)")


if __name__ == "__main__":
    main()
