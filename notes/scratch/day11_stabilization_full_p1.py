"""
Day 4's k* stabilization finding (mean k*=2.17 on the 100-problem
dev subset, notes/2026-08-21.md), rerun at the FULL 500-problem P1
scale. Pure local analysis -- P1 is fully generated, no new network
calls beyond the (already-fetched-elsewhere, cheap) gold answers.
Same algorithm as `day4_stabilization.py`, ported to read from the
real `PoolStore` instead of the scratch JSONL pool.
"""

import json
import sys
import time
import urllib.request
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")

from marginal_token.answers.equivalence import check_equivalent  # noqa: E402
from marginal_token.answers.extraction import extract_answer  # noqa: E402
from marginal_token.answers.taxonomy import FailureStatus  # noqa: E402
from marginal_token.pools.store import PoolStore  # noqa: E402

POOL_ROOT = "results/pools"
N = 32


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


def score_row(pool, gold):
    row = []
    for sample in pool.samples:
        extraction = extract_answer(sample.text, finish_reason=sample.finish_reason)
        if extraction.status != FailureStatus.OK:
            row.append((None, None))
            continue
        eq = check_equivalent(prediction=extraction.value, gold=gold)
        row.append((str(extraction.value), eq.equivalent))
    return row


def majority_at(row, k):
    votes = Counter(val for val, eq in row[:k] if val is not None)
    if not votes:
        return None, False
    top_key, top_count = votes.most_common(1)[0]
    tied = sum(1 for v in votes.values() if v == top_count) > 1
    if tied:
        return None, True
    return top_key, False


def pass_at_k_unbiased(n, c, k):
    import math
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def main():
    pool_meta = []
    for jsonl_path in sorted(Path(POOL_ROOT).glob("*/*.jsonl")):
        pool_id = jsonl_path.parent.name
        problem_id = jsonl_path.stem.replace("__", "/")
        if problem_id.isdigit():
            continue  # P2/OlympiadBench
        pool_meta.append((pool_id, problem_id))
    assert len(pool_meta) == 500, f"expected 500, got {len(pool_meta)}"

    gold = fetch_all_gold([pid for _, pid in pool_meta])
    print(f"fetched {len(gold)} gold answers")

    store = PoolStore(POOL_ROOT)
    results = []  # (problem_id, k_star, final_correct, status)
    pass_at_32_vals = []

    for pool_id, problem_id in pool_meta:
        pool = store.load(pool_id, problem_id, "math500", "qwen3.5-4b", "policy_primary")
        assert len(pool) == 32
        row = score_row(pool, gold[problem_id])

        c = sum(1 for _, eq in row if eq)
        pass_at_32_vals.append(pass_at_k_unbiased(32, c, 32))

        final_key, final_tied = majority_at(row, 32)
        if final_tied or final_key is None:
            results.append((problem_id, None, None, "tie_or_no_votes"))
            continue

        k_star = None
        for k in range(1, 33):
            key_k, tied_k = majority_at(row, k)
            if not tied_k and key_k == final_key:
                if k_star is None:
                    k_star = k
            else:
                k_star = None
        assert k_star is not None, f"{problem_id}: bug -- final majority never stabilizes against itself"

        final_correct = next(eq for val, eq in row if val == final_key)
        results.append((problem_id, k_star, final_correct, "ok"))

    print(f"\nmean pass@32 (ceiling) across all 500: {np.mean(pass_at_32_vals):.4f}")

    tie_problems = [r for r in results if r[3] == "tie_or_no_votes"]
    valid = [r for r in results if r[3] == "ok"]
    print(f"{len(valid)}/500 problems have a well-defined final N=32 majority; "
          f"{len(tie_problems)} excluded (tie or no valid votes at k=32)")

    ks = np.array([r[1] for r in valid])
    correct_mask = np.array([r[2] for r in valid])

    print(f"\nOverall mean k* (stabilization point): {ks.mean():.2f} (median {np.median(ks):.1f}, "
          f"min {ks.min()}, max {ks.max()})")
    print(f"Mean k* where final majority is CORRECT   (n={correct_mask.sum()}): "
          f"{ks[correct_mask].mean():.2f} (median {np.median(ks[correct_mask]):.1f})")
    print(f"Mean k* where final majority is INCORRECT (n={(~correct_mask).sum()}): "
          f"{ks[~correct_mask].mean():.2f} (median {np.median(ks[~correct_mask]):.1f})")

    n_le_8 = int((ks <= 8).sum())
    print(f"\n{n_le_8}/{len(valid)} ({n_le_8/len(valid):.1%}) stabilize by k*<=8 (well under the N=32 budget)")

    print("\n(Cross-check against the dev-100 subset finding, notes/2026-08-21.md: "
          "mean k*=2.17, median 1.0, 63/75 stabilize at k*=1. This is the FULL 500-problem "
          "P1 pool -- not expected to match exactly, but should be directionally consistent "
          "if the underlying finding generalizes beyond the original 100-problem sample.)")


if __name__ == "__main__":
    main()
