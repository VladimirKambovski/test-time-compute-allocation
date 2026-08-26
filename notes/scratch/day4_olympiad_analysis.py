"""
G1 fallback check on OlympiadBench-A: reuses day4_analysis.py's scoring
logic, pointed at the Olympiad pool instead of MATH-500's. See
notes/2026-08-21.md for why max_tokens=4096 was used here specifically.

Note: 2/30 problems (1606: 0/32 samples, 1833: 12/32 samples) hit
persistent HTTP 500 errors during generation, unrelated to the timeout/
concurrency issues seen elsewhere -- excluded automatically since they
don't reach the N=32 threshold. G1 here runs over the 28 problems with
complete pools, not 30 -- exactly the kind of small-sample noise the
user asked to see the bootstrap CI for.
"""
import json
import sys
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, "src")
from marginal_token.answers.equivalence import check_equivalent
from marginal_token.answers.extraction import extract_answer
from marginal_token.answers.taxonomy import FailureStatus

sys.path.insert(0, "notes/scratch")
from day4_analysis import deterministic_action_values, maj_at_k_mc, pass_at_k_unbiased

POOL_PATH = "notes/scratch/day4_olympiad_pool.jsonl"
K_VALUES = [1, 2, 4, 8, 16, 32]
RNG = np.random.default_rng(20260821)


def load_pool():
    by_problem = defaultdict(list)
    with open(POOL_PATH) as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                by_problem[rec["problem_id"]].append(rec)
    for pid in by_problem:
        by_problem[pid].sort(key=lambda r: r["sample_idx"])
    return by_problem


def score_samples(by_problem):
    scored = {}
    for pid, recs in by_problem.items():
        row = []
        for rec in recs:
            extraction = extract_answer(rec["model_prediction"], finish_reason=rec.get("finish_reason"))
            if extraction.status != FailureStatus.OK:
                row.append((extraction.status.value, None, None))
                continue
            eq = check_equivalent(prediction=extraction.value, gold=rec["gold_answer"])
            row.append(("ok", str(extraction.value), eq.equivalent))
        scored[pid] = row
    return scored


def main():
    by_problem = load_pool()
    n_problems = len(by_problem)
    sample_counts = {pid: len(v) for pid, v in by_problem.items()}
    complete = [pid for pid, n in sample_counts.items() if n >= 32]
    print(f"Loaded {n_problems} problems with any data; {len(complete)} have a complete N=32 pool")
    incomplete = {pid: n for pid, n in sample_counts.items() if n < 32}
    if incomplete:
        print(f"Excluded (incomplete): {incomplete}")

    scored = score_samples(by_problem)

    all_statuses = [s for row in scored.values() for (s, _, _) in row]
    status_counts = Counter(all_statuses)
    total = len(all_statuses)
    print("\n=== Extraction status rates (all generated samples, incl. incomplete problems) ===")
    for status, count in status_counts.most_common():
        print(f"  {status}: {count}/{total} = {count/total:.1%}")

    print("\n=== maj@k / pass@k curve, complete-pool problems only (n={}) ===".format(len(complete)))
    for k in K_VALUES:
        pass_vals, maj_vals = [], []
        for pid in complete:
            row = scored[pid]
            n = len(row)
            c = sum(1 for status, val, eq in row if status == "ok" and eq)
            pass_vals.append(pass_at_k_unbiased(n, c, k))
            maj = maj_at_k_mc(row, k, n_resamples=100)
            if maj is not None:
                maj_vals.append(maj)
        if pass_vals:
            print(f"  k={k:2d}: pass@k={np.mean(pass_vals):.3f}  maj@k={np.mean(maj_vals):.3f}")

    print(f"\n=== Gate G1 on OlympiadBench-A at N=32 (n={len(complete)} problems) ===")
    stop_arr, sample_arr, select_arr = [], [], []
    for pid in complete:
        row = scored[pid][:32]
        s, sa, se = deterministic_action_values(row)
        stop_arr.append(s)
        sample_arr.append(sa)
        select_arr.append(se)

    stop_arr = np.array(stop_arr)
    sample_arr = np.array(sample_arr)
    select_arr = np.array(select_arr)
    oracle_correct = stop_arr | sample_arr | select_arr
    best_fixed_acc = max(stop_arr.mean(), sample_arr.mean())
    oracle_acc = oracle_correct.mean()
    gap = (oracle_acc - best_fixed_acc) * 100

    diff = oracle_correct.astype(float) - np.maximum(stop_arr, sample_arr).astype(float)
    n = len(diff)
    boot_idx = RNG.integers(0, n, size=(10000, n))
    boot_gaps = diff[boot_idx].mean(axis=1) * 100
    ci_lo, ci_hi = np.percentile(boot_gaps, [2.5, 97.5])

    print(f"  n_problems: {n}")
    print(f"  STOP accuracy:   {stop_arr.mean():.3f}")
    print(f"  SAMPLE accuracy: {sample_arr.mean():.3f}")
    print(f"  SELECT ceiling (pass@32): {select_arr.mean():.3f}")
    print(f"  Oracle accuracy: {oracle_acc:.3f}")
    print(f"  Best fixed policy accuracy: {best_fixed_acc:.3f}")
    print(f"  GAP: {gap:.2f}pp, 95% bootstrap CI: [{ci_lo:.2f}, {ci_hi:.2f}]")
    print(f"  G1 accept (>=8pp, CI excludes 0): "
          f"{'PASS' if gap >= 8 and ci_lo > 0 else 'CHECK -- does not clear threshold'}")


if __name__ == "__main__":
    main()
