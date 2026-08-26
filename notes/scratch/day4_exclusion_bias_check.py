"""
Are the 25 problems excluded from the k* stabilization analysis (tie or
no clear majority at k=32) disproportionately the same problems where
SAMPLE beats STOP -- i.e., where G1's small gap actually concentrates?
No new generation -- reuses the existing N=32 pool.
"""
import json
import sys
from collections import defaultdict

sys.path.insert(0, "src")
from marginal_token.answers.equivalence import check_equivalent
from marginal_token.answers.extraction import extract_answer
from marginal_token.answers.taxonomy import FailureStatus

sys.path.insert(0, "notes/scratch")
from day4_analysis import deterministic_action_values
from day4_stabilization import load_pool, majority_at, score_row

POOL_PATH = "notes/scratch/day4_pool.jsonl"


def main():
    by_problem = load_pool()  # already filters to sample_idx < 32
    assert len(by_problem) == 100

    excluded, kept = [], []
    gap_relevant = []  # SAMPLE correct AND STOP incorrect -- where SAMPLE beats STOP
    reverse_relevant = []  # STOP correct AND SAMPLE incorrect -- the rarer opposite case
    neither = []

    for pid, recs in by_problem.items():
        row_str = score_row(recs)  # [(val_str, eq_or_None), ...]
        _, final_tied = majority_at(row_str, 32)
        is_excluded = final_tied or majority_at(row_str, 32)[0] is None

        # deterministic_action_values needs the (status, val, eq) triple format,
        # not score_row's (val, eq) -- recompute directly for STOP/SAMPLE correctness.
        row_full = []
        for rec in recs:
            extraction = extract_answer(rec["model_prediction"], finish_reason=rec.get("finish_reason"))
            if extraction.status != FailureStatus.OK:
                row_full.append((extraction.status.value, None, None))
                continue
            eq = check_equivalent(prediction=extraction.value, gold=rec["gold_answer"])
            row_full.append(("ok", str(extraction.value), eq.equivalent))

        stop_correct, sample_correct, select_correct = deterministic_action_values(row_full)

        if is_excluded:
            excluded.append(pid)
        else:
            kept.append(pid)

        if sample_correct and not stop_correct:
            gap_relevant.append(pid)
        elif stop_correct and not sample_correct:
            reverse_relevant.append(pid)
        else:
            neither.append(pid)

    print(f"Total: {len(by_problem)}  Excluded (tie/no-vote): {len(excluded)}  Kept: {len(kept)}")
    print(f"Gap-relevant overall (SAMPLE correct, STOP incorrect): {len(gap_relevant)}/100")
    print(f"Reverse-relevant (STOP correct, SAMPLE incorrect): {len(reverse_relevant)}/100")
    print()

    excluded_set = set(excluded)
    gap_in_excluded = [p for p in gap_relevant if p in excluded_set]
    gap_in_kept = [p for p in gap_relevant if p not in excluded_set]
    reverse_in_excluded = [p for p in reverse_relevant if p in excluded_set]
    reverse_in_kept = [p for p in reverse_relevant if p not in excluded_set]

    print(f"Gap-relevant problems (SAMPLE beats STOP): {len(gap_relevant)} total")
    print(f"  -> in EXCLUDED set: {len(gap_in_excluded)}/{len(excluded)} excluded problems = "
          f"{len(gap_in_excluded)/len(excluded):.1%} of excluded" if excluded else "  (no excluded)")
    print(f"  -> in KEPT set:     {len(gap_in_kept)}/{len(kept)} kept problems = "
          f"{len(gap_in_kept)/len(kept):.1%} of kept")
    print(f"  Base rate overall: {len(gap_relevant)}/100 = {len(gap_relevant)/100:.1%}")
    print()
    print(f"Reverse-relevant problems (STOP beats SAMPLE): {len(reverse_relevant)} total")
    print(f"  -> in EXCLUDED set: {len(reverse_in_excluded)}/{len(excluded)} = "
          f"{len(reverse_in_excluded)/len(excluded):.1%} of excluded" if excluded else "")
    print(f"  -> in KEPT set:     {len(reverse_in_kept)}/{len(kept)} = "
          f"{len(reverse_in_kept)/len(kept):.1%} of kept")

    print()
    print("Gap-relevant problem IDs:", gap_relevant)
    print("Excluded problem IDs:", excluded)
    print("Overlap (gap-relevant AND excluded):", gap_in_excluded)


if __name__ == "__main__":
    main()
