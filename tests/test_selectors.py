"""
Day 8: selectors + the canonical `accuracy()` aggregator. Unit tests use
synthetic votes; the real-pool cross-check at the bottom replays Day 4/5's
already-known, already-reported numbers (0.730 plain majority, B3's
"zero individual flips" finding) through the productionized code as a
strong regression check -- not a new analysis, a correctness proof that
this module reproduces what was already trusted.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import pytest

from marginal_token.selectors.basic import MajorityResult, VoteEntry, accuracy, plain_majority
from marginal_token.selectors.prm_based import WeightedVoteEntry, prm_argmax, prm_weighted_majority

REPO_ROOT = Path(__file__).resolve().parent.parent
DAY4_POOL_PATH = REPO_ROOT / "notes/scratch/day4_pool.jsonl"
DAY5_PRM_SCORES_PATH = REPO_ROOT / "notes/scratch/day5_full_pool_prm_scores.jsonl"


# --- accuracy(): the resolved tie/no-vote convention -------------------


def test_accuracy_counts_a_correct_win_toward_the_numerator():
    results = [MajorityResult(winning_key="4", is_tie=False, is_correct=True)]
    assert accuracy(results) == 1.0


def test_accuracy_counts_an_incorrect_win_toward_denominator_only():
    results = [MajorityResult(winning_key="4", is_tie=False, is_correct=False)]
    assert accuracy(results) == 0.0


def test_accuracy_counts_a_tie_as_incorrect_not_excluded():
    """The exact convention Day 5's B3 bug required fixing: a tie
    (`is_correct=None`) counts toward the denominator as an incorrect
    result -- it does NOT shrink the denominator by being dropped.
    """
    results = [
        MajorityResult(winning_key="4", is_tie=False, is_correct=True),
        MajorityResult(winning_key=None, is_tie=True, is_correct=None),
        MajorityResult(winning_key=None, is_tie=True, is_correct=None),
    ]
    assert accuracy(results) == pytest.approx(1 / 3)


def test_accuracy_counts_a_no_vote_as_incorrect_too():
    results = [
        MajorityResult(winning_key="4", is_tie=False, is_correct=True),
        MajorityResult(winning_key=None, is_tie=False, is_correct=None),  # nobody produced a usable answer
    ]
    assert accuracy(results) == 0.5


def test_accuracy_raises_on_empty_input_rather_than_a_meaningless_0_over_0():
    with pytest.raises(ValueError):
        accuracy([])


# --- plain_majority ------------------------------------------------------


def test_plain_majority_picks_the_plurality_and_reports_its_correctness():
    votes = [VoteEntry("4", True), VoteEntry("4", True), VoteEntry("5", False)]
    result = plain_majority(votes)
    assert result.winning_key == "4"
    assert result.is_correct is True
    assert result.is_tie is False


def test_plain_majority_reports_a_genuine_tie_honestly():
    votes = [VoteEntry("4", True), VoteEntry("5", False)]
    result = plain_majority(votes)
    assert result.is_tie is True
    assert result.winning_key is None
    assert result.is_correct is None


def test_plain_majority_ignores_samples_with_no_usable_answer():
    votes = [VoteEntry("4", True), VoteEntry(None, None), VoteEntry(None, None)]
    result = plain_majority(votes)
    assert result.winning_key == "4"
    assert result.is_correct is True


# --- prm_weighted_majority / prm_argmax ---------------------------------


def test_prm_weighted_majority_can_overturn_a_plain_plurality():
    # "5" has more raw votes, but "4"'s single vote carries more total
    # PRM weight -- PRM-weighted majority must pick "4", unlike plain
    # majority which would pick "5".
    votes = [
        WeightedVoteEntry("5", False, 0.1),
        WeightedVoteEntry("5", False, 0.1),
        WeightedVoteEntry("4", True, 0.9),
    ]
    result = prm_weighted_majority(votes)
    assert result.winning_key == "4"
    assert result.is_correct is True


def test_prm_weighted_majority_excludes_unscored_samples_from_both_vote_and_weight():
    votes = [
        WeightedVoteEntry("4", True, 0.9),
        WeightedVoteEntry("5", False, None),  # never scored (e.g. step_segmentation_failed)
    ]
    result = prm_weighted_majority(votes)
    assert result.winning_key == "4"  # the unscored "5" contributes nothing, not weight=0


def test_prm_weighted_majority_reports_a_weight_tie_honestly():
    votes = [WeightedVoteEntry("4", True, 0.5), WeightedVoteEntry("5", False, 0.5)]
    result = prm_weighted_majority(votes)
    assert result.is_tie is True
    assert result.is_correct is None


def test_prm_argmax_picks_the_single_highest_scored_sample_no_voting():
    votes = [
        WeightedVoteEntry("5", False, 0.4),
        WeightedVoteEntry("5", False, 0.5),
        WeightedVoteEntry("4", True, 0.9),  # highest score wins outright, regardless of vote count
    ]
    result = prm_argmax(votes)
    assert result.winning_key == "4"
    assert result.is_correct is True


def test_prm_argmax_no_usable_candidates_is_honest_not_a_default():
    votes = [WeightedVoteEntry(None, None, None), WeightedVoteEntry("5", False, None)]
    result = prm_argmax(votes)
    assert result.winning_key is None
    assert result.is_correct is None


# --- real-pool cross-check: reproduce Day 4/5's already-known numbers --


def _load_day4_pool() -> dict[str, list[dict]]:
    by_problem = defaultdict(list)
    with open(DAY4_POOL_PATH) as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                if rec["sample_idx"] < 32:  # N=32, matching the primary G1/B3 scale
                    by_problem[rec["problem_id"]].append(rec)
    return by_problem


def _load_day5_prm_scores() -> dict[tuple[str, int], float | None]:
    scores = {}
    with open(DAY5_PRM_SCORES_PATH) as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                scores[(d["problem_id"], d["sample_idx"])] = d.get("mean_reward")
    return scores


@pytest.mark.skipif(
    not (DAY4_POOL_PATH.exists() and DAY5_PRM_SCORES_PATH.exists()),
    reason="Day 4/5 real-pool artifacts not present in this checkout -- this cross-check needs "
           "notes/scratch/day4_pool.jsonl and day5_full_pool_prm_scores.jsonl, which are real "
           "generated/scored data, not committed fixtures.",
)
def test_selectors_reproduce_day4_day5_headline_numbers_on_the_real_pool():
    """Not a mocked check -- this replays the actual 100-problem, N=32
    MATH-500 pool (`notes/scratch/day4_pool.jsonl`) and the actual PRM
    scores from Day 5's full-pool run
    (`notes/scratch/day5_full_pool_prm_scores.jsonl`) through
    `plain_majority`/`prm_weighted_majority` + `accuracy()`, and asserts
    the result matches the already-reported, already-trusted numbers
    exactly: plain majority = 0.730 (notes/2026-08-21.md's G1 baseline),
    PRM-weighted majority = 0.730 too with zero individual flips
    (notes/2026-08-23.md's B3 reproduction). If this module disagreed
    with those numbers, that would mean the productionized code has the
    same kind of bug Day 5's scratch script did -- this is the
    regression test for exactly that.
    """
    import sys as _sys

    sys.path.insert(0, str(REPO_ROOT / "src"))
    from marginal_token.answers.equivalence import check_equivalent
    from marginal_token.answers.extraction import extract_answer
    from marginal_token.answers.taxonomy import FailureStatus

    by_problem = _load_day4_pool()
    prm_scores = _load_day5_prm_scores()
    assert len(by_problem) == 100

    plain_results: list[MajorityResult] = []
    weighted_results: list[MajorityResult] = []

    for pid, recs in by_problem.items():
        recs = sorted(recs, key=lambda r: r["sample_idx"])
        plain_votes: list[VoteEntry] = []
        weighted_votes: list[WeightedVoteEntry] = []
        for rec in recs:
            extraction = extract_answer(rec["model_prediction"], finish_reason=rec.get("finish_reason"))
            if extraction.status != FailureStatus.OK:
                plain_votes.append(VoteEntry(None, None))
                weighted_votes.append(WeightedVoteEntry(None, None, None))
                continue
            eq = check_equivalent(prediction=extraction.value, gold=rec["gold_answer"])
            key = str(extraction.value)
            plain_votes.append(VoteEntry(key, eq.equivalent))
            weight = prm_scores.get((pid, rec["sample_idx"]))
            weighted_votes.append(WeightedVoteEntry(key, eq.equivalent, weight))

        plain_results.append(plain_majority(plain_votes))
        weighted_results.append(prm_weighted_majority(weighted_votes))

    plain_acc = accuracy(plain_results)
    weighted_acc = accuracy(weighted_results)

    assert plain_acc == pytest.approx(0.730), f"plain majority = {plain_acc}, expected 0.730"
    assert weighted_acc == pytest.approx(0.730), f"PRM-weighted majority = {weighted_acc}, expected 0.730"

    n_flips = sum(1 for p, w in zip(plain_results, weighted_results) if p.is_correct != w.is_correct)
    assert n_flips == 0, "Day 5's B3 finding was zero individual-problem flips -- this must still hold"
