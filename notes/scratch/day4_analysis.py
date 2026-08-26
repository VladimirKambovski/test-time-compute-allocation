"""
Day 4 analysis: maj@k/pass@k curves + Gate G1, from notes/scratch/day4_pool.jsonl.

Two distinct estimators, deliberately not conflated:
- The maj@k/pass@k CURVE (for the B1/B2 sanity-check comparison) uses
  proper unbiased estimators (Codex/HumanEval-style combinatorial pass@k;
  Monte Carlo subsampling for maj@k) -- this is about the population-level
  shape of the curve, and benefits from averaging over many possible
  subsets rather than one arbitrary ordering.
- Gate G1 uses the SPECIFIC, deterministic realization a real system
  would actually see: the first 4 generated samples ARE the probe (STOP),
  all 32 ARE the full budget (SAMPLE/SELECT) -- no resampling tricks,
  because G1 asks "would this specific deployed pipeline show headroom,"
  not "what's the theoretical curve."

Known simplification, stated rather than silently assumed: majority-vote
clustering of WRONG answers uses a canonical string key (str() of the
math_verify-parsed value) rather than full pairwise equivalence-checking
between every pair of wrong answers. Cheap and usually right; two
differently-formatted-but-equivalent wrong answers could in principle be
undercounted as separate clusters. Not expected to flip conclusions given
G1's headroom is expected to be large (per the pre-check simulation),
but noted for the record.
"""
import json
import math
import sys
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, "src")
from marginal_token.answers.equivalence import check_equivalent
from marginal_token.answers.extraction import extract_answer
from marginal_token.answers.taxonomy import FailureStatus

POOL_PATH = "notes/scratch/day4_pool.jsonl"
K_VALUES = [1, 2, 4, 8, 16, 32, 64]
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
    """Returns {problem_id: [(status, value_str_or_None, equivalent_bool_or_None), ...]}
    in sample_idx order.
    """
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


def pass_at_k_unbiased(n, c, k):
    """Codex/HumanEval unbiased pass@k estimator."""
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def maj_at_k_mc(row, k, n_resamples=200):
    """Monte Carlo estimate of P(majority vote over a random k-subset is
    correct), using canonical string-key clustering for wrong answers.
    """
    n = len(row)
    if k > n:
        return None
    successes = 0
    for _ in range(n_resamples):
        idx = RNG.choice(n, size=k, replace=False)
        subset = [row[i] for i in idx]
        votes = Counter()
        correct_key = None
        for status, val, eq in subset:
            if status != "ok":
                continue  # non-extracted samples don't vote
            votes[val] += 1
            if eq:
                correct_key = val
        if not votes:
            continue  # nobody voted -- counts as a miss
        top_key, top_count = votes.most_common(1)[0]
        # strict plurality required; ties broken against (conservative)
        tied = sum(1 for v in votes.values() if v == top_count) > 1
        if not tied and top_key == correct_key:
            successes += 1
    return successes / n_resamples


def deterministic_action_values(row):
    """The G1-relevant, single-realization values: STOP=maj(first 4),
    SAMPLE=maj(all), SELECT=pass(all) -- no resampling.
    """
    n = len(row)
    probe = row[:4]
    votes = Counter()
    correct_key = None
    for status, val, eq in probe:
        if status != "ok":
            continue
        votes[val] += 1
        if eq:
            correct_key = val
    if votes:
        top_key, top_count = votes.most_common(1)[0]
        tied = sum(1 for v in votes.values() if v == top_count) > 1
        stop_correct = (not tied) and top_key == correct_key
    else:
        stop_correct = False

    votes_all = Counter()
    correct_key_all = None
    for status, val, eq in row:
        if status != "ok":
            continue
        votes_all[val] += 1
        if eq:
            correct_key_all = val
    if votes_all:
        top_key, top_count = votes_all.most_common(1)[0]
        tied = sum(1 for v in votes_all.values() if v == top_count) > 1
        sample_correct = (not tied) and top_key == correct_key_all
    else:
        sample_correct = False

    select_correct = any(eq for status, val, eq in row if status == "ok" and eq)

    return stop_correct, sample_correct, select_correct


def main():
    by_problem = load_pool()
    n_problems = len(by_problem)
    sample_counts = [len(v) for v in by_problem.values()]
    print(f"Loaded {n_problems} problems, sample counts: min={min(sample_counts)} max={max(sample_counts)} "
          f"(expect 32 once generation completes)")

    scored = score_samples(by_problem)

    # --- extraction-failure / truncation rates ---
    all_statuses = [s for row in scored.values() for (s, _, _) in row]
    status_counts = Counter(all_statuses)
    total = len(all_statuses)
    print("\n=== Extraction status rates ===")
    for status, count in status_counts.most_common():
        print(f"  {status}: {count}/{total} = {count/total:.1%}")

    # --- pass@k / maj@k curves ---
    print("\n=== maj@k / pass@k curve (unbiased estimators, averaged over problems) ===")
    for k in K_VALUES:
        pass_vals, maj_vals = [], []
        for pid, row in scored.items():
            n = len(row)
            if n < k:
                continue
            c = sum(1 for status, val, eq in row if status == "ok" and eq)
            pass_vals.append(pass_at_k_unbiased(n, c, k))
            maj = maj_at_k_mc(row, k, n_resamples=100)
            if maj is not None:
                maj_vals.append(maj)
        if pass_vals:
            print(f"  k={k:2d}: pass@k={np.mean(pass_vals):.3f}  maj@k={np.mean(maj_vals):.3f}  (n_problems={len(pass_vals)})")

    # --- Gate G1, at every N we have data for (prefix consistency + the real question) ---
    max_n = min(len(row) for row in scored.values())
    for target_n in sorted({8, 16, 32, 64} & {n for n in (8, 16, 32, 64) if n <= max_n}):
        print(f"\n=== Gate G1 at N={target_n}: oracle-over-{{STOP,SAMPLE,SELECT-ceiling}} vs best-fixed-{{STOP,SAMPLE}} ===")
        stop_arr, sample_arr, select_arr = [], [], []
        for pid, row in scored.items():
            if len(row) < target_n:
                continue
            s, sa, se = deterministic_action_values(row[:target_n])
            stop_arr.append(s)
            sample_arr.append(sa)
            select_arr.append(se)

        if len(stop_arr) < 100:
            print(f"  Only {len(stop_arr)}/100 problems have complete N={target_n} pools -- "
                  f"partial run, G1 not final yet.")
            continue
        stop_arr = np.array(stop_arr)
        sample_arr = np.array(sample_arr)
        select_arr = np.array(select_arr)
        oracle_correct = stop_arr | sample_arr | select_arr
        best_fixed_acc = max(stop_arr.mean(), sample_arr.mean())
        oracle_acc = oracle_correct.mean()
        gap = (oracle_acc - best_fixed_acc) * 100

        diff = oracle_correct.astype(float) - np.maximum(stop_arr, sample_arr).astype(float)
        # paired bootstrap CI on the gap (percentile method, 10k resamples per brief.md §19)
        n = len(diff)
        boot_idx = RNG.integers(0, n, size=(10000, n))
        boot_gaps = diff[boot_idx].mean(axis=1) * 100
        ci_lo, ci_hi = np.percentile(boot_gaps, [2.5, 97.5])

        print(f"  STOP accuracy:   {stop_arr.mean():.3f}")
        print(f"  SAMPLE accuracy: {sample_arr.mean():.3f}")
        print(f"  SELECT ceiling (pass@{target_n}): {select_arr.mean():.3f}")
        print(f"  Oracle accuracy: {oracle_acc:.3f}")
        print(f"  Best fixed policy accuracy: {best_fixed_acc:.3f}")
        print(f"  GAP: {gap:.2f}pp, 95% bootstrap CI: [{ci_lo:.2f}, {ci_hi:.2f}]")
        print(f"  G1 accept (>=8pp, CI excludes 0): "
              f"{'PASS' if gap >= 8 and ci_lo > 0 else 'CHECK -- does not clear threshold'}")


if __name__ == "__main__":
    main()
