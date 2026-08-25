"""
PRM-weighted majority and PRM-argmax -- the two MUST selectors that need
a real per-sample PRM score, implementing action A2 (SELECT).

Held out of `basic.py` until two conditions were both met: Day 5's real
PRM integration existed (`src/marginal_token/scoring/`), and the
tie/no-vote accuracy-aggregation convention was resolved (see
`basic.py`'s docstring and `accuracy()`). Both are done as of Day 8.

Both selectors return the same `MajorityResult` shape `basic.py`'s
`plain_majority` does, specifically so `basic.py::accuracy()` scores all
selectors identically -- see that function's docstring for why a
per-selector inconsistency here would be exactly Day 5's B3 bug again.

Reminder carried over from Day 4/5, not re-litigated here: SELECT was
narrowed out of the controller's ACTION space (oracle win rate 1-3%,
`notes/2026-08-22.md`/`notes/2026-08-23.md`), but PRM score remains a
planned predictor *feature* (docs/brief.md §16) independent of that --
these selectors exist for B3/B-series baseline reproduction and the
predictor's feature set, not because SELECT was reinstated as a
controller action.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from marginal_token.selectors.basic import MajorityResult


@dataclass
class WeightedVoteEntry:
    """One sample's contribution: its canonical answer key (or None if it
    didn't produce a usable answer), whether that answer is correct, and
    its PRM weight (e.g. `mean_reward` from `scoring.pipeline.PRMScore`;
    None if the sample was never scored -- `step_segmentation_failed` or
    `prm_score_missing`, per the closed taxonomy).
    """

    answer_key: str | None
    is_correct: bool | None
    weight: float | None


def prm_weighted_majority(votes: list[WeightedVoteEntry]) -> MajorityResult:
    """A2/SELECT's real scoring rule: group samples by answer key, sum
    each group's PRM weight (not a plain vote count), pick the
    highest-total-weight key. A sample with no usable answer OR no
    usable PRM weight contributes to neither a vote nor a weight --
    excluded, never silently treated as weight=0 (invariant #6/#7: a
    missing score is not the same claim as a score of zero).
    """
    weights: dict[str, float] = defaultdict(float)
    correctness: dict[str, bool | None] = {}
    for v in votes:
        if v.answer_key is None or v.weight is None:
            continue
        weights[v.answer_key] += v.weight
        correctness[v.answer_key] = v.is_correct
    if not weights:
        return MajorityResult(winning_key=None, is_tie=False, is_correct=None)
    top_key, top_weight = max(weights.items(), key=lambda kv: kv[1])
    tied = sum(1 for w in weights.values() if w == top_weight) > 1
    if tied:
        return MajorityResult(winning_key=None, is_tie=True, is_correct=None)
    return MajorityResult(winning_key=top_key, is_tie=False, is_correct=correctness[top_key])


def prm_argmax(votes: list[WeightedVoteEntry]) -> MajorityResult:
    """The other selector B3 names ("PRM best-of-N", docs/brief.md line
    101): skip voting entirely, the single highest-PRM-scored sample
    wins outright. A sample with no usable answer or no usable weight is
    never a candidate.
    """
    candidates = [v for v in votes if v.answer_key is not None and v.weight is not None]
    if not candidates:
        return MajorityResult(winning_key=None, is_tie=False, is_correct=None)
    top_weight = max(v.weight for v in candidates)
    tied_candidates = [v for v in candidates if v.weight == top_weight]
    if len(tied_candidates) > 1:
        return MajorityResult(winning_key=None, is_tie=True, is_correct=None)
    winner = tied_candidates[0]
    return MajorityResult(winning_key=winner.answer_key, is_tie=False, is_correct=winner.is_correct)
