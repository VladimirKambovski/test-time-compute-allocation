"""
Day 13: temperature ablation analysis. Paired comparison, same 21
MATH-500 problems (difficulty band 1, the moderately-hard band), same
policy/backend, decode identical except temperature (0.8, the frozen
setting used for every sample generated project-wide, vs. 1.0, the
ablation arm). Tests whether the frozen G1/SELECT findings are an
artifact of the one temperature used everywhere.

Baseline arm (temp=0.8): reused directly from the existing P1 pool --
no regeneration needed, these 21 problems are a subset of P1's 500.
Ablation arm (temp=1.0): notes/scratch/day13_temp_ablation_ids.json's
pool, configs/pools/day13-temp-ablation-t1.0.yaml.

n=21 is well below P1's full-scale n=500 -- treat this as a directional
diagnostic per the brief's own honesty rule (n>=5 permits a bootstrap CI,
but a wide CI here is expected and not itself a null result).
"""

import json
import sys

sys.path.insert(0, "src")

import numpy as np  # noqa: E402

from marginal_token.answers.equivalence import check_equivalent  # noqa: E402
from marginal_token.answers.extraction import extract_answer  # noqa: E402
from marginal_token.answers.taxonomy import FailureStatus  # noqa: E402
from marginal_token.controller.oracle_labels import oracle_action_label  # noqa: E402
from marginal_token.evaluation.stats import paired_bootstrap_bca  # noqa: E402
from marginal_token.generation.run_sweeps import fetch_math500_problems  # noqa: E402
from marginal_token.pools.store import PoolStore, compute_pool_id  # noqa: E402

POOL_ROOT = "results/pools"
IDS_FILE = "notes/scratch/day13_temp_ablation_ids.json"


def fetch_gold_for_ids(ids):
    # fetch_math500_problems gives (id, problem_text), not gold -- reuse
    # the same gold-fetch pattern as day12/day13-select scripts instead.
    import time
    import urllib.request

    wanted = set(ids)
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


def load_pool_for(store, pid, benchmark_id, temperature):
    pool_id = compute_pool_id(
        policy_ref="qwen3.5-4b", backend_ref="policy_primary", benchmark_id=benchmark_id,
        problem_id=pid, temperature=temperature, top_p=0.95, max_tokens=1024, seed=None, n=32,
    )
    return store.load(pool_id, pid, benchmark_id, "qwen3.5-4b", "policy_primary")


def eval_arm(store, ids, gold, benchmark_id, temperature):
    stop_correct, sample_correct, select_correct, actions = [], [], [], []
    for pid in ids:
        pool = load_pool_for(store, pid, benchmark_id, temperature)
        assert len(pool) == 32, f"{pid} at temp={temperature}: expected 32 samples, got {len(pool)}"
        result = oracle_action_label(pool.samples, gold[pid])
        stop_correct.append(result.stop_correct)
        sample_correct.append(result.sample_correct)
        select_correct.append(result.select_correct)
        actions.append(result.action)
    stop_correct = np.array(stop_correct, dtype=float)
    sample_correct = np.array(sample_correct, dtype=float)
    oracle_correct = np.array([s or a for s, a in zip(stop_correct.astype(bool), sample_correct.astype(bool))], dtype=float)
    # oracle here is the STOP/SAMPLE/SELECT union (== select_correct, since
    # select_correct is "any correct", a strict superset) -- keep both for
    # transparency.
    select_ceiling = np.array(select_correct, dtype=float)
    best_fixed = np.maximum(stop_correct, sample_correct)
    gap = select_ceiling - best_fixed
    return {
        "stop_acc": stop_correct.mean(), "sample_acc": sample_correct.mean(),
        "oracle_ceiling": select_ceiling.mean(), "best_fixed": best_fixed.mean(),
        "gap_pp": 100 * gap.mean(), "gap_per_problem": gap,
        "select_only_rate": sum(1 for a in actions if a == "select") / len(actions),
        "actions": actions,
    }


def main():
    ids = json.load(open(IDS_FILE))
    print(f"{len(ids)} problems in the temperature ablation")

    gold = fetch_gold_for_ids(ids)
    missing = [pid for pid in ids if pid not in gold]
    if missing:
        print(f"WARNING: {len(missing)} problems missing gold answers, dropping: {missing}")
        ids = [pid for pid in ids if pid in gold]

    store = PoolStore(POOL_ROOT)

    print("\n-- baseline arm: temp=0.8 (existing P1 data, same problems) --")
    base = eval_arm(store, ids, gold, "math500", 0.8)
    for k in ("stop_acc", "sample_acc", "oracle_ceiling", "best_fixed", "gap_pp", "select_only_rate"):
        print(f"  {k}: {base[k]:.4f}" if "rate" in k or "acc" in k or "ceiling" in k or "fixed" in k else f"  {k}: {base[k]:.2f}")

    print("\n-- ablation arm: temp=1.0 (new data, same problems) --")
    abl = eval_arm(store, ids, gold, "math500-temp-ablation", 1.0)
    for k in ("stop_acc", "sample_acc", "oracle_ceiling", "best_fixed", "gap_pp", "select_only_rate"):
        print(f"  {k}: {abl[k]:.4f}" if "rate" in k or "acc" in k or "ceiling" in k or "fixed" in k else f"  {k}: {abl[k]:.2f}")

    diff = abl["gap_per_problem"] - base["gap_per_problem"]
    print(f"\npaired per-problem gap difference (temp1.0 - temp0.8), n={len(ids)}:")
    print(f"  mean: {diff.mean():.4f} ({100*diff.mean():.2f}pp)")
    boot = paired_bootstrap_bca(diff, seed=20260825)
    print(f"  BCa 95% CI on mean diff: [{boot.ci_lo:.4f}, {boot.ci_hi:.4f}] "
          f"({100*boot.ci_lo:.2f}pp, {100*boot.ci_hi:.2f}pp)")
    excludes_zero = boot.ci_lo > 0 or boot.ci_hi < 0
    print(f"  CI excludes zero: {excludes_zero}")

    print(f"\nSELECT-only action count: temp0.8={sum(1 for a in base['actions'] if a=='select')}/{len(ids)}, "
          f"temp1.0={sum(1 for a in abl['actions'] if a=='select')}/{len(ids)}")

    print("\n(n=21, a small diagnostic sample from one difficulty band -- not a "
          "replacement for the full-scale G1/SELECT results, a directional check "
          "on whether temperature is a confound.)")


if __name__ == "__main__":
    main()
