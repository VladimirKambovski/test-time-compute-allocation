"""
Day 13: quick scoping check triggered by the temperature ablation --
does `_majority_correct`/`_any_correct`'s silent exclusion of failed
extractions (length_truncated etc.) from the vote produce a "trivial
majority among 1-3 survivors" pattern broadly across P1, or is it
confined to the hard/truncation-heavy tail (band 1/band 0)? Streams P1
only, one pool at a time (memory-safe).
"""
import gc
import json
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, "src")

from marginal_token.answers.extraction import extract_answer  # noqa: E402
from marginal_token.answers.equivalence import check_equivalent  # noqa: E402
from marginal_token.answers.taxonomy import FailureStatus  # noqa: E402
from marginal_token.evaluation.stats import difficulty_bands  # noqa: E402
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

    pass_at_1 = {}
    n_ok = {}
    n_processed = 0
    for pool_id, pid in math_meta:
        if pid not in gold:
            continue
        pool = store.load(pool_id, pid, "math500", "qwen3.5-4b", "policy_primary")
        if len(pool) != 32:
            continue  # skip N=64-extended dupes etc, keep this a clean N=32 comparison
        c = 0
        ok = 0
        for s in pool.samples:
            ext = extract_answer(s.text, finish_reason=s.finish_reason)
            if ext.status == FailureStatus.OK:
                ok += 1
                if check_equivalent(prediction=ext.value, gold=gold[pid]).equivalent:
                    c += 1
        pass_at_1[pid] = c / 32
        n_ok[pid] = ok
        n_processed += 1
        if n_processed % 100 == 0:
            gc.collect()
            print(f"  {n_processed} processed")

    print(f"\ntotal P1 problems processed (N=32 only): {n_processed}")
    bands = difficulty_bands(pass_at_1, n_bands=5)

    from collections import defaultdict
    by_band_nok = defaultdict(list)
    for pid, b in bands.items():
        by_band_nok[b].append(n_ok[pid])

    print("\nband (0=hardest, 4=easiest): mean successfully-extracted samples /32, frac problems with <=3 survivors")
    for b in range(5):
        vals = by_band_nok[b]
        frac_trivial = sum(1 for v in vals if v <= 3) / len(vals)
        print(f"  band {b}: n={len(vals)}, mean_n_ok={sum(vals)/len(vals):.2f}, frac_<=3_survivors={frac_trivial:.2%}")


if __name__ == "__main__":
    main()
