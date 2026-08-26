"""
Auto-run after the overnight generation finishes: compares the 24
next-hardest MATH-500 problems before (1024 tokens) vs after (4096
tokens), same pattern as tonight's 6-problem check, so there's a
finished, readable summary waiting in the morning rather than raw pool
files needing separate analysis.
"""
import json
import sys
import urllib.request

sys.path.insert(0, "src")

from marginal_token.answers.extraction import extract_answer  # noqa: E402
from marginal_token.answers.taxonomy import FailureStatus  # noqa: E402
from marginal_token.controller.oracle_labels import oracle_action_label  # noqa: E402
from marginal_token.pools.store import PoolStore, compute_pool_id  # noqa: E402

IDS_FILE = "notes/scratch/day13_overnight_maxtokens_ids.json"


def main():
    ids = json.load(open(IDS_FILE))
    print(f"{len(ids)} problems in the overnight max_tokens extension")

    gold = {}
    for offset in range(0, 500, 100):
        url = (f"https://datasets-server.huggingface.co/rows?dataset=HuggingFaceH4/MATH-500"
               f"&config=default&split=test&offset={offset}&length=100")
        d = json.load(urllib.request.urlopen(url, timeout=30))
        for r in d["rows"]:
            if r["row"]["unique_id"] in ids:
                gold[r["row"]["unique_id"]] = r["row"]["answer"]

    store = PoolStore("results/pools")
    flipped_away_from_abstain = 0
    action_counts_after = {"stop": 0, "sample": 0, "select": 0, "abstain": 0}
    n_ok_before_total, n_ok_after_total = 0, 0

    for pid in ids:
        if pid not in gold:
            print(f"=== {pid} ===  SKIPPED: no gold answer found")
            continue
        print(f"=== {pid} ===  gold: {gold[pid]}")
        results = {}
        for label, benchmark_id, mt in [("1024tok", "math500", 1024), ("4096tok", "math500-overnight-maxtokens", 4096)]:
            pool_id = compute_pool_id(policy_ref="qwen3.5-4b", backend_ref="policy_primary", benchmark_id=benchmark_id,
                                        problem_id=pid, temperature=0.8, top_p=0.95, max_tokens=mt, seed=None, n=32)
            pool = store.load(pool_id, pid, benchmark_id, "qwen3.5-4b", "policy_primary")
            n_ok = sum(1 for s in pool.samples if extract_answer(s.text, finish_reason=s.finish_reason).status == FailureStatus.OK)
            result = oracle_action_label(pool.samples, gold[pid])
            results[label] = (n_ok, result.action)
            print(f"  {label}: n_ok={n_ok}/32, action={result.action}")

        before_action = results["1024tok"][1]
        after_action = results["4096tok"][1]
        n_ok_before_total += results["1024tok"][0]
        n_ok_after_total += results["4096tok"][0]
        action_counts_after[after_action] += 1
        if before_action == "abstain" and after_action != "abstain":
            flipped_away_from_abstain += 1
        print()

    n = len([pid for pid in ids if pid in gold])
    print("=== SUMMARY ===")
    print(f"n={n} problems (all pass@1=0.000, mostly 0-3/32 extractions at 1024 tokens)")
    print(f"flipped away from ABSTAIN at 4096 tokens: {flipped_away_from_abstain}/{n} ({100*flipped_away_from_abstain/n:.1f}%)")
    print(f"action distribution at 4096 tokens: {action_counts_after}")
    print(f"mean successful extractions: before={n_ok_before_total/n:.2f}/32, after={n_ok_after_total/n:.2f}/32")
    print(f"\ncombined with tonight's hand-tested 6 (6/6 flipped, 5/6 to STOP, 1/6 to SAMPLE):")
    print(f"total systematic sample size: {n + 6} of band 0's ~100-151 problems")


if __name__ == "__main__":
    main()
