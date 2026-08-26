"""
Day 5: reproduce B3 (docs/brief.md line 101 -- "PRM-weighted majority and
PRM best-of-N ... PRM >= maj at small N, margin consistent with published
~1.4%") on the full N=32 Day-4 pool (100 MATH-500 problems), now that
day5_score_full_pool.py has scored it with the primary PRM
(double_newline segmentation, confirmed by G3: AUROC=0.9934).

Three selectors compared, all at N=32, same pool used for Day 4's G1
plain-majority number (0.730) so this is a direct, matched comparison:

- plain_majority: plurality vote by answer key (ties excluded from the
  numerator the same way day4_analysis.py/selectors/basic.py do -- a
  tied top count has no single winner to score).
- prm_weighted_majority: group samples by answer key, weight each
  sample's vote by its PRM mean_reward (not a plain count), pick the
  answer key with the highest total weight. This is A2/SELECT's actual
  scoring rule.
- prm_best_of_n: skip voting entirely, take the single highest-mean_reward
  sample in the pool and use its answer directly (the other selector
  named in B3).

Known simplification, carried over from day4_analysis.py's own stated
one: answer-key clustering uses a canonical string key (str() of the
math_verify-parsed value), not full pairwise equivalence between every
candidate pair. Same caveat applies here as there.

Samples with no usable answer (`is_correct is None`: no_boxed_answer /
length_truncated / extraction_ambiguous / equivalence_timeout) or with a
missing PRM score (`prm_status != "ok"`, e.g. step_segmentation_failed)
never get a vote and never get a weight for that individual sample.

**Tie/no-vote convention -- deliberately NOT `selectors/basic.py`'s.**
First draft of this script excluded a problem from the denominator
entirely on a tied or empty vote (`selectors/basic.py::plain_majority`'s
convention: tie -> `is_correct=None`). That produced plain-majority
accuracy = 0.9733 on 75/100 problems -- wildly inconsistent with Day 4's
own frozen G1 baseline for the exact same metric on the exact same pool
(SAMPLE accuracy = 0.730 on all 100, notes/2026-08-21.md). Root cause:
`notes/scratch/day4_analysis.py::deterministic_action_values` (the
actual G1 gate computation, already reported) uses a DIFFERENT
convention -- a tied or empty vote counts as INCORRECT, always included
in a denominator of 100, never excluded. Excluding ties instead
disproportionately drops exactly the hard, low-consensus problems (which
correlate with the majority vote being wrong), which is why it inflated
accuracy so sharply. Fixed here to match `day4_analysis.py`'s convention
exactly, since that's the one already reported as the project's headline
SAMPLE-accuracy number -- a B3 comparison against a plain-majority figure
computed under a *different* convention would not be a valid comparison,
even though both functions are named "plain majority."

This is a real, unresolved inconsistency between `selectors/basic.py`
(the Day-8-designated production selector, built ahead of schedule) and
the actual gate computation -- flagged in today's notes rather than
silently patched into `selectors/basic.py`, since that's a Day 8 design
decision (it feeds the oracle action labels) with downstream
consequences beyond today's B3 reproduction.
"""
import json
import sys
from collections import defaultdict

sys.path.insert(0, "src")
from marginal_token.answers.equivalence import check_equivalent  # noqa: E402
from marginal_token.answers.extraction import extract_answer  # noqa: E402
from marginal_token.answers.taxonomy import FailureStatus  # noqa: E402

POOL_PATH = "notes/scratch/day4_pool.jsonl"
PRM_PATH = "notes/scratch/day5_full_pool_prm_scores.jsonl"
N_SAMPLES = 32
PUBLISHED_MARGIN = 0.014  # B3's cited ~1.4% (2502.06703)


def load_pool():
    by_problem = defaultdict(list)
    with open(POOL_PATH) as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                if rec["sample_idx"] < N_SAMPLES:
                    by_problem[rec["problem_id"]].append(rec)
    for pid in by_problem:
        by_problem[pid].sort(key=lambda r: r["sample_idx"])
    return by_problem


def load_prm_scores():
    """{(problem_id, sample_idx): mean_reward or None}"""
    scores = {}
    with open(PRM_PATH) as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                key = (d["problem_id"], d["sample_idx"])
                scores[key] = d.get("mean_reward")
    return scores


def build_votes(pid, recs, prm_scores):
    """One entry per sample: (answer_key_or_None, is_correct_or_None, mean_reward_or_None)."""
    votes = []
    for rec in recs:
        extraction = extract_answer(rec["model_prediction"], finish_reason=rec.get("finish_reason"))
        if extraction.status != FailureStatus.OK:
            votes.append((None, None, prm_scores.get((pid, rec["sample_idx"]))))
            continue
        eq = check_equivalent(prediction=extraction.value, gold=rec["gold_answer"])
        key = str(extraction.value)
        votes.append((key, eq.equivalent, prm_scores.get((pid, rec["sample_idx"]))))
    return votes


def plain_majority_correct(votes):
    """Tie or no-vote -> counted as INCORRECT, always in a denominator of
    100. Matches day4_analysis.py::deterministic_action_values exactly --
    see module docstring for why this, not selectors/basic.py's
    exclude-on-tie convention, is the one that makes this a valid B3
    comparison against the already-reported 0.730 baseline.
    """
    counts = defaultdict(int)
    correctness = {}
    for key, is_correct, _ in votes:
        if key is None:
            continue
        counts[key] += 1
        correctness[key] = is_correct
    if not counts:
        return False
    top_key, top_count = max(counts.items(), key=lambda kv: kv[1])
    tied = sum(1 for v in counts.values() if v == top_count) > 1
    if tied:
        return False
    return bool(correctness[top_key])


def prm_weighted_majority_correct(votes):
    """Same tie/no-vote convention as plain_majority_correct, applied to
    PRM-mean_reward-weighted vote totals instead of raw counts.
    """
    weights = defaultdict(float)
    correctness = {}
    for key, is_correct, weight in votes:
        if key is None or weight is None:
            continue
        weights[key] += weight
        correctness[key] = is_correct
    if not weights:
        return False
    top_key, top_weight = max(weights.items(), key=lambda kv: kv[1])
    tied = sum(1 for w in weights.values() if w == top_weight) > 1
    if tied:
        return False
    return bool(correctness[top_key])


def prm_best_of_n_correct(votes):
    """No voting -- single highest-mean_reward sample wins outright. No
    usable candidate -> incorrect (same "no signal = wrong" convention).
    """
    candidates = [(weight, is_correct) for key, is_correct, weight in votes if key is not None and weight is not None]
    if not candidates:
        return False
    _, is_correct = max(candidates, key=lambda c: c[0])
    return bool(is_correct)


def main():
    by_problem = load_pool()
    prm_scores = load_prm_scores()
    n_problems = len(by_problem)
    print(f"{n_problems} problems, {len(prm_scores)} PRM-scored samples loaded")

    correct = {"plain_majority": [], "prm_weighted_majority": [], "prm_best_of_n": []}
    tie_or_novote = {"plain_majority": [], "prm_weighted_majority": [], "prm_best_of_n": []}

    for pid, recs in by_problem.items():
        votes = build_votes(pid, recs, prm_scores)

        pm = plain_majority_correct(votes)
        correct["plain_majority"].append(pm)

        pw = prm_weighted_majority_correct(votes)
        correct["prm_weighted_majority"].append(pw)

        pb = prm_best_of_n_correct(votes)
        correct["prm_best_of_n"].append(pb)

        # Diagnostic only, not used to change the denominator: which
        # problems hit the tie/no-vote branch (scored as incorrect above).
        counts = defaultdict(int)
        for key, _, _ in votes:
            if key is not None:
                counts[key] += 1
        if counts:
            top_count = max(counts.values())
            if sum(1 for v in counts.values() if v == top_count) > 1:
                tie_or_novote["plain_majority"].append(pid)
        else:
            tie_or_novote["plain_majority"].append(pid)

    print(f"\n{'selector':<24}{'n':>6}{'n_correct':>12}{'accuracy':>12}")
    accs = {}
    for name, results in correct.items():
        acc = sum(results) / n_problems
        accs[name] = acc
        print(f"{name:<24}{n_problems:>6}{sum(results):>12}{acc:>12.4f}")

    print(f"\nplain_majority tie/no-vote problems (scored as incorrect, per "
          f"day4_analysis.py's convention -- {len(tie_or_novote['plain_majority'])}/100): "
          f"{tie_or_novote['plain_majority']}")

    margin = accs["prm_weighted_majority"] - accs["plain_majority"]
    print(f"\nPRM-weighted majority vs plain majority margin: {margin*100:+.2f}pp "
          f"(published ~{PUBLISHED_MARGIN*100:.1f}pp)")
    best_of_n_margin = accs["prm_best_of_n"] - accs["plain_majority"]
    print(f"PRM best-of-N vs plain majority margin:          {best_of_n_margin*100:+.2f}pp")

    b3_condition = accs["prm_weighted_majority"] >= accs["plain_majority"]
    print(f"\nB3 literal condition (PRM-weighted >= plain majority): {'PASS' if b3_condition else 'FAIL'}")
    print(f"Cross-check against Day 4's frozen G1 baseline: plain_majority here = {accs['plain_majority']:.4f}, "
          f"day4_analysis.py's SAMPLE accuracy (N=32) = 0.730 -- {'MATCH' if abs(accs['plain_majority'] - 0.730) < 1e-6 else 'MISMATCH, investigate before trusting anything else in this script'}")


if __name__ == "__main__":
    main()
