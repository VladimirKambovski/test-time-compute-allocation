"""G1 analysis on the weaker-policy (Qwen3.5-2B) pool. Reuses day4_analysis.py's
scoring/G1 machinery, pointed at the weaker-policy pool instead of the 4B one."""
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

POOL_PATH = "notes/scratch/day4_weaker_policy_pool.jsonl"
RNG = np.random.default_rng(20260822)


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
    assert len(by_problem) == 100, f"expected 25, got {len(by_problem)}"
    scored = score_samples(by_problem)

    all_statuses = [s for row in scored.values() for (s, _, _) in row]
    status_counts = Counter(all_statuses)
    total = len(all_statuses)
    print("=== Extraction status rates (Qwen3.5-2B) ===")
    for status, count in status_counts.most_common():
        print(f"  {status}: {count}/{total} = {count/total:.1%}")

    print("\n=== maj@k / pass@k curve (Qwen3.5-2B) ===")
    for k in (1, 2, 4, 8, 16, 32):
        pass_vals, maj_vals = [], []
        for pid, row in scored.items():
            n = len(row)
            c = sum(1 for status, val, eq in row if status == "ok" and eq)
            pass_vals.append(pass_at_k_unbiased(n, c, k))
            maj = maj_at_k_mc(row, k, n_resamples=100)
            if maj is not None:
                maj_vals.append(maj)
        print(f"  k={k:2d}: pass@k={np.mean(pass_vals):.3f}  maj@k={np.mean(maj_vals):.3f}")

    print("\n=== Gate G1 on Qwen3.5-2B, N=32 (n=100 problems) ===")
    stop_arr, sample_arr, select_arr = [], [], []
    action_counts = {"stop": 0, "sample_only": 0, "select_only": 0, "abstain": 0}
    for pid, row in scored.items():
        row32 = row[:32]
        s, sa, se = deterministic_action_values(row32)
        stop_arr.append(s)
        sample_arr.append(sa)
        select_arr.append(se)
        if s:
            action_counts["stop"] += 1
        elif sa:
            action_counts["sample_only"] += 1
        elif se:
            action_counts["select_only"] += 1
        else:
            action_counts["abstain"] += 1

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

    print(f"  STOP accuracy:   {stop_arr.mean():.3f}")
    print(f"  SAMPLE accuracy: {sample_arr.mean():.3f}")
    print(f"  SELECT ceiling (pass@32): {select_arr.mean():.3f}")
    print(f"  Oracle accuracy: {oracle_acc:.3f}")
    print(f"  Best fixed policy accuracy: {best_fixed_acc:.3f}")
    print(f"  GAP: {gap:.2f}pp, 95% bootstrap CI: [{ci_lo:.2f}, {ci_hi:.2f}]")
    print(f"  G1 accept (>=8pp, CI excludes 0): "
          f"{'PASS' if gap >= 8 and ci_lo > 0 else 'CHECK -- does not clear threshold'}")
    print(f"  Oracle-action distribution: {action_counts} (n=100)")

    # k* stabilization, same method as the 4B analysis
    print("\n=== k* stabilization (Qwen3.5-2B) ===")
    ks = []
    excluded = 0
    for pid, row in scored.items():
        row_str = [(val, eq) for status, val, eq in row]
        votes = Counter(v for v, eq in row_str if v is not None)
        if not votes:
            excluded += 1
            continue
        top_key, top_count = votes.most_common(1)[0]
        tied = sum(1 for v in votes.values() if v == top_count) > 1
        if tied:
            excluded += 1
            continue
        final_key = top_key
        k_star = None
        for k in range(1, 33):
            sub = row_str[:k]
            v = Counter(val for val, eq in sub if val is not None)
            if not v:
                k_star = None
                continue
            tk, tc = v.most_common(1)[0]
            t = sum(1 for c in v.values() if c == tc) > 1
            if not t and tk == final_key:
                if k_star is None:
                    k_star = k
            else:
                k_star = None
        ks.append(k_star)
    print(f"  {len(ks)}/100 well-defined ({excluded} excluded for tie/no-vote)")
    if ks:
        ks_arr = np.array(ks)
        print(f"  mean k* = {ks_arr.mean():.2f}, median = {np.median(ks_arr):.1f}")


if __name__ == "__main__":
    main()
