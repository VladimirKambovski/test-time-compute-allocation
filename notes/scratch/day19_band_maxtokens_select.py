"""
Day 19, mentor-directed: select 6 problems from each of bands 1-4 (band
0 already has a real, tested n=30 sample from earlier tonight -- reused,
not regenerated) for a stratified cross-band max_tokens=4096 check.
Tests whether raising the budget changes the AGGREGATE G1 gap across the
full difficulty spectrum, not just the already-tested extreme (band 0).

Canonical-only enumeration throughout (compute_pool_id, never a glob) --
same discipline as every script since the duplicate-pool bug was found.
Bands computed on the full P1+P2 canonical 754-problem set, matching
the official Day-12 band definitions exactly (not a re-derivation).
"""
import gc
import json
import sys
import time
import urllib.request
from collections import defaultdict

sys.path.insert(0, "src")

from marginal_token.answers.equivalence import check_equivalent  # noqa: E402
from marginal_token.answers.extraction import extract_answer  # noqa: E402
from marginal_token.answers.taxonomy import FailureStatus  # noqa: E402
from marginal_token.evaluation.stats import difficulty_bands  # noqa: E402
from marginal_token.pools.store import PoolStore, compute_pool_id  # noqa: E402

ALREADY_TESTED_BAND0 = {
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
}


def fetch_math500_gold_all():
    found = {}
    offset = 0
    while offset < 500:
        url = (f"https://datasets-server.huggingface.co/rows?dataset=HuggingFaceH4/MATH-500"
               f"&config=default&split=test&offset={offset}&length=100")
        d = json.load(urllib.request.urlopen(url, timeout=30))
        for r in d["rows"]:
            row = r["row"]
            found[row["unique_id"]] = row["answer"]
        offset += 100
    return found


def fetch_olympiad_gold_all(wanted_ids):
    wanted = set(wanted_ids)
    found = {}
    offset = 0
    while offset < 674 and len(found) < len(wanted):
        url = (f"https://datasets-server.huggingface.co/rows?dataset=Hothan/OlympiadBench"
               f"&config=OE_TO_maths_en_COMP&split=train&offset={offset}&length=100")
        d = json.load(urllib.request.urlopen(url, timeout=30))
        for r in d["rows"]:
            row = r["row"]
            row_id = str(row["id"])
            if row_id in wanted and not row.get("is_multiple_answer") and not row.get("error") and row.get("final_answer"):
                found[row_id] = row["final_answer"][0]
        offset += 100
    return found


def main():
    print("fetching gold answers...", flush=True)
    math_gold = fetch_math500_gold_all()
    oly_ids = [str(x) for x in json.load(open("configs/benchmarks/data/olympiad-a-ids.json"))]
    oly_gold = fetch_olympiad_gold_all(oly_ids)
    print(f"got {len(math_gold)} MATH-500, {len(oly_gold)}/{len(oly_ids)} OlympiadBench-A", flush=True)

    store = PoolStore("results/pools")
    pass_at_1 = {}
    n_processed = 0

    for pid, gold in math_gold.items():
        pool_id = compute_pool_id(policy_ref="qwen3.5-4b", backend_ref="policy_primary", benchmark_id="math500",
                                    problem_id=pid, temperature=0.8, top_p=0.95, max_tokens=1024, seed=None, n=32)
        pool = store.load(pool_id, pid, "math500", "qwen3.5-4b", "policy_primary")
        assert len(pool) == 32
        c = sum(1 for s in pool.samples if extract_answer(s.text, finish_reason=s.finish_reason).status == FailureStatus.OK
                and check_equivalent(prediction=extract_answer(s.text, finish_reason=s.finish_reason).value, gold=gold).equivalent)
        pass_at_1[f"math500:{pid}"] = c / 32
        n_processed += 1
        if n_processed % 150 == 0:
            gc.collect()
            print(f"  {n_processed} processed", flush=True)

    for pid in oly_ids:
        if pid not in oly_gold:
            continue
        gold = oly_gold[pid]
        pool_id = compute_pool_id(policy_ref="qwen3.5-4b", backend_ref="policy_primary", benchmark_id="olympiad-a",
                                    problem_id=pid, temperature=0.8, top_p=0.95, max_tokens=1024, seed=None, n=32)
        pool = store.load(pool_id, pid, "olympiad-a", "qwen3.5-4b", "policy_primary")
        assert len(pool) == 32
        c = sum(1 for s in pool.samples if extract_answer(s.text, finish_reason=s.finish_reason).status == FailureStatus.OK
                and check_equivalent(prediction=extract_answer(s.text, finish_reason=s.finish_reason).value, gold=gold).equivalent)
        pass_at_1[f"olympiad-a:{pid}"] = c / 32
        n_processed += 1
        if n_processed % 150 == 0:
            gc.collect()
            print(f"  {n_processed} processed", flush=True)

    print(f"\ntotal usable problems: {n_processed}", flush=True)
    bands = difficulty_bands(pass_at_1, n_bands=5)

    by_band = defaultdict(list)
    for key, b in bands.items():
        benchmark, pid = key.split(":", 1)
        by_band[b].append((benchmark, pid, pass_at_1[key]))

    selected = {"math500": [], "olympiad-a": []}
    for band in (1, 2, 3, 4):
        candidates = sorted(by_band[band], key=lambda t: (t[2], t[1]))
        candidates = [c for c in candidates if not (c[0] == "math500" and c[1] in ALREADY_TESTED_BAND0)]
        stride = max(1, len(candidates) // 6)
        picks = candidates[::stride][:6]
        print(f"band {band}: {len(by_band[band])} total, picked {len(picks)}: {[(p[0], p[1], round(p[2],3)) for p in picks]}")
        for benchmark, pid, _ in picks:
            selected[benchmark].append(pid)

    json.dump(selected["math500"], open("notes/scratch/day19_band_maxtokens_math500_ids.json", "w"), indent=2)
    json.dump(selected["olympiad-a"], open("notes/scratch/day19_band_maxtokens_olympiad_ids.json", "w"), indent=2)
    print(f"\nwrote {len(selected['math500'])} MATH-500 ids, {len(selected['olympiad-a'])} OlympiadBench-A ids")


if __name__ == "__main__":
    main()
