"""Plain majority and oracle pass@k -- the two MUST selectors that don't
need a PRM. Pure functions over (samples, gold), no side effects.

This is genuinely shared infrastructure: the exact same majority-vote and
pass@k logic already validated in the Day 4 k* stabilization analysis
(notes/2026-08-21.md) and the golden-200/G1 pipeline
(notes/scratch/day4_analysis.py) is what's productionized here -- this
commit is NOT new analysis toward the pending H1/H4 mentor-input
question, it's packaging already-exercised logic as a reusable module.
PRM-weighted majority and PRM-argmax now live in `prm_based.py` -- they
were held out of this module until Day 5's real PRM integration existed
and until the tie-handling question below was resolved. Both conditions
are now met.

RESOLVED (Day 8, was a TODO): `plain_majority` returns `is_correct=None`
on a tied or empty vote -- an honest per-problem signal ("no single
answer to check"), kept as-is below. What was actually inconsistent
project-wide was how callers AGGREGATE a list of these results into an
accuracy number: an earlier draft excluded `is_correct=None` entries
from the denominator, which disagreed with
`notes/scratch/day4_analysis.py::deterministic_action_values` (the
script that produced every already-reported headline SAMPLE-accuracy
number, e.g. 0.730 on the Day-4 100-problem N=32 pool) -- that script
counts a tie/empty-vote as INCORRECT, denominator never shrinks. Found
2026-08-23 reproducing B3 (`notes/scratch/day5_reproduce_b3.py`):
excluding ties inflated an intermediate accuracy check to 0.9733/75,
because excluding ties disproportionately drops the hard, low-consensus
problems that correlate with the majority vote being wrong. Full
writeup: `notes/2026-08-23.md`.

**Fix: `accuracy()` below is now the ONE canonical way any selector's
per-problem results get turned into an accuracy number** -- it treats
`is_correct in (None, False)` identically (both count toward the
denominator, neither counts toward the numerator), matching
`deterministic_action_values` exactly. `plain_majority`,
`prm_weighted_majority`, and `prm_argmax` (in `prm_based.py`) all return
the same `MajorityResult` shape specifically so they can all be scored
through this one function -- a per-selector aggregation inconsistency
would silently invalidate any selector-vs-selector comparison the same
way it did here.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Any


@dataclass
class VoteEntry:
    """One sample's contribution to a vote: its canonical answer key (or
    None if it didn't produce a usable answer -- e.g. length_truncated),
    and whether that answer is equivalent to gold.
    """

    answer_key: str | None
    is_correct: bool | None


@dataclass
class MajorityResult:
    winning_key: str | None  # None if tied or nobody voted
    is_tie: bool
    is_correct: bool | None  # None if no winner to check


def plain_majority(votes: list[VoteEntry]) -> MajorityResult:
    """A1 SAMPLE's selector: plurality vote among samples that produced a
    usable answer. A strict plurality is required -- a tie is reported as
    a tie, never silently broken toward "correct" or toward an arbitrary
    winner.
    """
    counts = Counter(v.answer_key for v in votes if v.answer_key is not None)
    if not counts:
        return MajorityResult(winning_key=None, is_tie=False, is_correct=None)
    top_key, top_count = counts.most_common(1)[0]
    tied = sum(1 for c in counts.values() if c == top_count) > 1
    if tied:
        return MajorityResult(winning_key=None, is_tie=True, is_correct=None)
    is_correct = next(v.is_correct for v in votes if v.answer_key == top_key)
    return MajorityResult(winning_key=top_key, is_tie=False, is_correct=is_correct)


def pass_at_k_unbiased(n: int, c: int, k: int) -> float:
    """The standard Codex/HumanEval unbiased pass@k estimator: probability
    that at least one of a random k-subset (without replacement) of the n
    samples is correct, given c of the n are correct.
    """
    if k > n:
        raise ValueError(f"k={k} cannot exceed n={n}")
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def oracle_pass_at_k(votes: list[VoteEntry], k: int) -> float:
    """Oracle pass@k ceiling over `votes` (treated as the full pool of
    n=len(votes) samples): ignores which answer is a plurality, only
    asks "does a correct answer exist anywhere in a random k-subset."
    This is the MUST oracle-ceiling selector (docs/brief.md §14) -- an
    upper bound on what any selector, however good, could achieve at
    budget k, not a claim about what plain majority or a real PRM
    actually achieves.
    """
    n = len(votes)
    c = sum(1 for v in votes if v.is_correct)
    return pass_at_k_unbiased(n, c, k)


def votes_from_samples(extracted: list[tuple[str | None, bool | None]]) -> list[VoteEntry]:
    """Convenience constructor from (answer_key, is_correct) tuples, e.g.
    straight from `marginal_token.answers.extraction`/`equivalence`
    output -- keeps the answers/ module's output shape decoupled from
    this module's internal VoteEntry type.
    """
    return [VoteEntry(answer_key=k, is_correct=c) for k, c in extracted]


def accuracy(results: list[MajorityResult]) -> float:
    """The one canonical way to turn a list of per-problem selector
    results into an accuracy number. A tied or empty vote
    (`is_correct=None`) counts as INCORRECT -- the denominator is always
    `len(results)`, never shrunk by excluding unresolved problems. This
    matches `notes/scratch/day4_analysis.py::deterministic_action_values`
    exactly (see this module's docstring for why that's the convention
    that had to win). Every selector in this module and in
    `prm_based.py` returns `MajorityResult`, specifically so they can all
    be scored through this one function -- never write a second,
    differently-behaved accuracy computation for a different selector.
    """
    if not results:
        raise ValueError("accuracy() needs at least one result -- an empty list has no denominator")
    return sum(1 for r in results if r.is_correct) / len(results)
