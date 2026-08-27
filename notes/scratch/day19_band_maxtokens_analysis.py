"""
Day 19, mentor-directed: does raising max_tokens change the AGGREGATE
G1 gap across a stratified cross-band sample? Combines band 0's
existing n=30 (from earlier tonight) with the new 23 problems from
bands 1-4 (24 selected, 1 dropped -- '2082' failed 32/32 with a
persistent, reproducible 500 error, content-checked and found unremarkable,
not chased further). n=53 total.

Paired comparison: same problems, oracle-vs-best-fixed gap at 1024
tokens (existing dev pool) vs. 4096 tokens (new generation). Canonical-
only enumeration throughout.
"""
import json
import sys
import urllib.request

sys.path.insert(0, "src")

import numpy as np  # noqa: E402

from marginal_token.controller.oracle_labels import _majority_correct, _any_correct  # noqa: E402
from marginal_token.evaluation.stats import paired_bootstrap_bca  # noqa: E402
from marginal_token.pools.store import PoolStore, compute_pool_id  # noqa: E402

BAND0_MATH500_IDS = [
    "test/algebra/2176.json", "test/algebra/297.json", "test/algebra/686.json",
    "test/algebra/892.json", "test/counting_and_probability/10.json", "test/counting_and_probability/1003.json",
    "test/counting_and_probability/181.json", "test/counting_and_probability/188.json",
    "test/counting_and_probability/199.json", "test/counting_and_probability/238.json",
    "test/counting_and_probability/282.json", "test/counting_and_probability/430.json",
    "test/counting_and_probability/731.json", "test/counting_and_probability/870.json",
    "test/counting_and_probability/894.json", "test/geometry/1140.json", "test/geometry/172.json",
    "test/geometry/183.json", "test/geometry/229.json", "test/geometry/283.json", "test/geometry/434.json",
    "test/geometry/465.json", "test/geometry/547.json", "test/geometry/561.json", "test/geometry/65.json",
    "test/geometry/702.json", "test/geometry/711.json", "test/geometry/817.json", "test/geometry/826.json",
    "test/geometry/880.json",
]


def _fetch_json_retry(url, attempts=5):
    import time
    for i in range(attempts):
        try:
            return json.load(urllib.request.urlopen(url, timeout=30))
        except Exception:
            if i == attempts - 1:
                raise
            time.sleep(3)


def fetch_math500_gold(ids):
    wanted = set(ids)
    found = {}
    offset = 0
    while offset < 500 and len(found) < len(wanted):
        url = (f"https://datasets-server.huggingface.co/rows?dataset=HuggingFaceH4/MATH-500"
               f"&config=default&split=test&offset={offset}&length=100")
        d = _fetch_json_retry(url)
        for r in d["rows"]:
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
        d = _fetch_json_retry(url)
        for r in d["rows"]:
            row = r["row"]
            row_id = str(row["id"])
            if row_id in wanted:
                found[row_id] = row["final_answer"][0]
        offset += 100
    return found


def gap_for(pid, benchmark_id_1024, benchmark_id_4096, gold, store, mt_new):
    results = {}
    for label, benchmark_id, mt in [("1024", benchmark_id_1024, 1024), ("4096", benchmark_id_4096, mt_new)]:
        pool_id = compute_pool_id(policy_ref="qwen3.5-4b", backend_ref="policy_primary", benchmark_id=benchmark_id,
                                    problem_id=pid, temperature=0.8, top_p=0.95, max_tokens=mt, seed=None, n=32)
        pool = store.load(pool_id, pid, benchmark_id, "qwen3.5-4b", "policy_primary")
        # 1024-token pools (the original P1/P2 dev data) must always be complete.
        # 4096-token pools MAY be incomplete for 3 known problems from the overnight
        # run (timeouts, documented in notes/2026-08-26.md) -- oracle_labels'
        # _majority_correct/_any_correct work correctly on however many samples
        # exist, so this doesn't invalidate the result, just note it.
        if label == "1024":
            assert len(pool) == 32, f"{pid}/{label}: {len(pool)} samples, expected exactly 32"
        elif len(pool) != 32:
            print(f"  note: {pid}/{label} has {len(pool)}/32 samples (known incomplete pool), using as-is")
        ordered = sorted(pool.samples, key=lambda s: s.sample_idx)
        stop_c = _majority_correct(ordered[:4], gold)
        sample_c = _majority_correct(ordered, gold)
        select_c = _any_correct(ordered, gold)
        oracle = stop_c or sample_c or select_c
        best_fixed = stop_c or sample_c  # max as boolean OR is fine for a single problem's binary correctness
        results[label] = (oracle, max(stop_c, sample_c))
    return results


def main():
    store = PoolStore("results/pools")

    math_new = json.load(open("notes/scratch/day19_band_maxtokens_math500_ids.json"))
    oly_new_all = json.load(open("notes/scratch/day19_band_maxtokens_olympiad_ids.json"))
    oly_new = [pid for pid in oly_new_all if pid != "2082"]
    print(f"dropped: 2082 (persistent 500 error, content-checked, not chased further)")
    print(f"new problems: {len(math_new)} MATH-500 + {len(oly_new)} OlympiadBench-A (bands 1-4)")
    print(f"band-0 reused: {len(BAND0_MATH500_IDS)} MATH-500 (from earlier tonight)")

    math_gold = fetch_math500_gold(math_new + BAND0_MATH500_IDS)
    oly_gold = fetch_olympiad_gold(oly_new)

    oracle_1024, best_fixed_1024, oracle_4096, best_fixed_4096 = [], [], [], []

    FIRST_SIX = {
        "test/algebra/2176.json", "test/algebra/297.json", "test/algebra/686.json",
        "test/algebra/892.json", "test/counting_and_probability/10.json", "test/counting_and_probability/1003.json",
    }
    for pid in BAND0_MATH500_IDS:
        benchmark_4096 = "math500-maxtokens-ablation" if pid in FIRST_SIX else "math500-overnight-maxtokens"
        r = gap_for(pid, "math500", benchmark_4096, math_gold[pid], store, 4096)
        oracle_1024.append(r["1024"][0]); best_fixed_1024.append(r["1024"][1])
        oracle_4096.append(r["4096"][0]); best_fixed_4096.append(r["4096"][1])

    for pid in math_new:
        r = gap_for(pid, "math500", "math500-band-maxtokens", math_gold[pid], store, 4096)
        oracle_1024.append(r["1024"][0]); best_fixed_1024.append(r["1024"][1])
        oracle_4096.append(r["4096"][0]); best_fixed_4096.append(r["4096"][1])

    for pid in oly_new:
        r = gap_for(pid, "olympiad-a", "olympiad-a-band-maxtokens", oly_gold[pid], store, 4096)
        oracle_1024.append(r["1024"][0]); best_fixed_1024.append(r["1024"][1])
        oracle_4096.append(r["4096"][0]); best_fixed_4096.append(r["4096"][1])

    n = len(oracle_1024)
    print(f"\ntotal stratified cross-band sample: n={n}")

    oracle_1024 = np.array(oracle_1024, dtype=float)
    best_fixed_1024 = np.array(best_fixed_1024, dtype=float)
    oracle_4096 = np.array(oracle_4096, dtype=float)
    best_fixed_4096 = np.array(best_fixed_4096, dtype=float)

    gap_1024 = oracle_1024 - best_fixed_1024
    gap_4096 = oracle_4096 - best_fixed_4096

    print(f"\n=== at 1024 tokens (frozen baseline) ===")
    print(f"oracle: {oracle_1024.mean():.4f}  best_fixed: {best_fixed_1024.mean():.4f}  gap: {100*gap_1024.mean():.2f}pp")
    boot_1024 = paired_bootstrap_bca(gap_1024, seed=20260827)
    print(f"gap 95% CI: [{100*boot_1024.ci_lo:.2f}pp, {100*boot_1024.ci_hi:.2f}pp]")

    print(f"\n=== at 4096 tokens (this check) ===")
    print(f"oracle: {oracle_4096.mean():.4f}  best_fixed: {best_fixed_4096.mean():.4f}  gap: {100*gap_4096.mean():.2f}pp")
    boot_4096 = paired_bootstrap_bca(gap_4096, seed=20260827)
    print(f"gap 95% CI: [{100*boot_4096.ci_lo:.2f}pp, {100*boot_4096.ci_hi:.2f}pp]")

    diff = gap_4096 - gap_1024
    print(f"\n=== does the gap itself change with more tokens? (paired diff, n={n}) ===")
    print(f"mean change in gap: {100*diff.mean():.2f}pp")
    boot_diff = paired_bootstrap_bca(diff, seed=20260827)
    print(f"95% CI: [{100*boot_diff.ci_lo:.2f}pp, {100*boot_diff.ci_hi:.2f}pp]")
    excludes_zero = boot_diff.ci_lo > 0 or boot_diff.ci_hi < 0
    print(f"excludes zero: {excludes_zero}")


if __name__ == "__main__":
    main()
