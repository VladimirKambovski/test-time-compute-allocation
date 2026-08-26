"""
Per MATH-500 problem: the smallest sample count k* such that the running
majority (over samples 0..k-1, in original generation order) equals the
final N=32 majority AND never changes again for any k in [k*, 32].
("Already matches" is read as stabilizes, not "happens to coincide once
then flips back" -- the more meaningful notion for "how many samples
before more sampling stops changing the answer.")

No new generation -- reuses notes/scratch/day4_pool.jsonl (already have
N=64, only using samples 0..31 to match the primary N=32 G1 test scale).

Ties (no strict plurality) are tracked separately, not silently broken,
at both the k=32 final-answer stage and at intermediate k -- a "tie"
never counts as a match against a non-tied final answer, and a
tied final answer excludes that problem from the average (can't define
"stabilizes to X" if there's no single X).
"""
import json
import sys
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, "src")
from marginal_token.answers.equivalence import check_equivalent
from marginal_token.answers.extraction import extract_answer
from marginal_token.answers.taxonomy import FailureStatus

POOL_PATH = "notes/scratch/day4_pool.jsonl"
N = 32  # match the primary G1 test scale, not the N=64 extension


def load_pool():
    by_problem = defaultdict(list)
    with open(POOL_PATH) as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                if rec["sample_idx"] < N:
                    by_problem[rec["problem_id"]].append(rec)
    for pid in by_problem:
        by_problem[pid].sort(key=lambda r: r["sample_idx"])
    return by_problem


def score_row(recs):
    row = []
    for rec in recs:
        extraction = extract_answer(rec["model_prediction"], finish_reason=rec.get("finish_reason"))
        if extraction.status != FailureStatus.OK:
            row.append((None, None))
            continue
        eq = check_equivalent(prediction=extraction.value, gold=rec["gold_answer"])
        row.append((str(extraction.value), eq.equivalent))
    return row


def majority_at(row, k):
    """Returns (winning_key, is_tie) for the plurality vote over row[:k].
    winning_key is None if nobody voted (all non-ok) or if tied.
    """
    votes = Counter(val for val, eq in row[:k] if val is not None)
    if not votes:
        return None, False
    top_key, top_count = votes.most_common(1)[0]
    tied = sum(1 for v in votes.values() if v == top_count) > 1
    if tied:
        return None, True
    return top_key, False


def main():
    by_problem = load_pool()
    assert len(by_problem) == 100, f"expected 100, got {len(by_problem)}"

    results = []  # (problem_id, k_star, final_correct, final_key_is_tie)
    for pid, recs in by_problem.items():
        assert len(recs) == 32, f"{pid} has {len(recs)} samples, expected 32"
        row = score_row(recs)
        final_key, final_tied = majority_at(row, 32)

        if final_tied or final_key is None:
            results.append((pid, None, None, "tie_or_no_votes"))
            continue

        # find smallest k* such that majority(row[:k]) == final_key for ALL k in [k*, 32]
        k_star = None
        for k in range(1, 33):
            key_k, tied_k = majority_at(row, k)
            if not tied_k and key_k == final_key:
                if k_star is None:
                    k_star = k
            else:
                k_star = None  # reset -- it changed again after this point
        assert k_star is not None, f"{pid}: final majority never matches itself at k=32? bug"

        # is the final majority answer correct? look up any row entry with that key
        final_correct = next(eq for val, eq in row if val == final_key)
        results.append((pid, k_star, final_correct, "ok"))

    tie_problems = [r for r in results if r[3] == "tie_or_no_votes"]
    valid = [r for r in results if r[3] == "ok"]
    print(f"{len(valid)}/100 problems have a well-defined final N=32 majority; "
          f"{len(tie_problems)} excluded (tie or no valid votes at k=32): "
          f"{[r[0] for r in tie_problems]}")

    ks = np.array([r[1] for r in valid])
    correct_mask = np.array([r[2] for r in valid])

    print(f"\nOverall mean k* (stabilization point): {ks.mean():.2f}  (median {np.median(ks):.1f}, "
          f"min {ks.min()}, max {ks.max()})")
    print(f"Mean k* where final majority is CORRECT   (n={correct_mask.sum()}): "
          f"{ks[correct_mask].mean():.2f}  (median {np.median(ks[correct_mask]):.1f})")
    print(f"Mean k* where final majority is INCORRECT (n={(~correct_mask).sum()}): "
          f"{ks[~correct_mask].mean():.2f}  (median {np.median(ks[~correct_mask]):.1f})")

    print("\nDistribution of k* (all valid problems):")
    for k in sorted(set(ks.tolist())):
        count = int((ks == k).sum())
        correct_count = int(((ks == k) & correct_mask).sum())
        print(f"  k*={k:2d}: {count:3d} problems ({correct_count} correct, {count-correct_count} incorrect)")


if __name__ == "__main__":
    main()
