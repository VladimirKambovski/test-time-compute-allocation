"""
Oracle action label a*(q,B): per problem, which action is actually
correct, given the FULL pool and gold answer.

**True 4-class label space, per docs/brief.md §16's literal spec**
("Primary 4-class over dev"): STOP, SAMPLE, SELECT, ABSTAIN. Revised
2026-08-23 -- an earlier version of this file hard-excluded SELECT from
the label space entirely, based on the empirical finding that it rarely
wins (notes/2026-08-22.md/2026-08-23.md). That was a real deviation
from the frozen spec's literal 4-class design, not just a report of the
finding -- reverted in favor of computing the TRUE 4-way label and
letting SELECT's near-zero rate show up as something the data
demonstrates, rather than something assumed true before the predictor
ever ran. This is strictly more informative (and no more expensive to
compute -- no new generation needed) than hard-coding the exclusion.

Convention: STOP wins if the k=4 probe's own majority is already
correct (cheapest correct action, no reason to spend more). Else SAMPLE
if the full-budget majority is correct. Else SELECT if a correct answer
exists ANYWHERE in the full pool even though the majority missed it --
this is exactly "the case a perfect PRM-based selector could have
rescued," matching the pass@N ceiling already used for the
oracle-pass@k selector (`selectors.basic.oracle_pass_at_k`) and the
"SELECT-only oracle win rate" already reported in
notes/2026-08-21.md/2026-08-22.md (1% at 4B, 3% at 2B). Else ABSTAIN
(no correct answer anywhere in the pool -- decline rather than
confidently return a wrong one). A tied or empty majority at the
STOP/SAMPLE stages counts as "not correct" at that stage, matching
`selectors.basic.accuracy()`'s resolved tie convention (Day 8).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Literal

from marginal_token.answers.equivalence import check_equivalent
from marginal_token.answers.extraction import extract_answer
from marginal_token.answers.taxonomy import FailureStatus
from marginal_token.backends.base import Sample

OracleAction = Literal["stop", "sample", "select", "abstain"]
PROBE_K = 4


@dataclass
class OracleLabelResult:
    action: OracleAction
    stop_correct: bool
    sample_correct: bool
    select_correct: bool  # True if ANY sample in the full pool is correct (the pass@N ceiling)


def _majority_correct(samples: list[Sample], gold: str) -> bool:
    counts: Counter[str] = Counter()
    correctness: dict[str, bool] = {}
    for s in samples:
        extraction = extract_answer(s.text, finish_reason=s.finish_reason)
        if extraction.status != FailureStatus.OK:
            continue
        key = str(extraction.value)
        counts[key] += 1
        if key not in correctness:
            eq = check_equivalent(prediction=extraction.value, gold=gold)
            correctness[key] = bool(eq.equivalent)
    if not counts:
        return False
    top_key, top_count = counts.most_common(1)[0]
    tied = sum(1 for c in counts.values() if c == top_count) > 1
    if tied:
        return False
    return correctness[top_key]


def _any_correct(samples: list[Sample], gold: str) -> bool:
    """The pass@N ceiling: does a correct answer exist anywhere in the
    pool, regardless of whether it's the majority? Answer-key clustering
    (not full pairwise re-verification of every sample) -- once one
    sample with a given canonical key is confirmed correct, every other
    sample sharing that exact key is correct too, so this never
    re-verifies the same key twice.
    """
    checked: dict[str, bool] = {}
    for s in samples:
        extraction = extract_answer(s.text, finish_reason=s.finish_reason)
        if extraction.status != FailureStatus.OK:
            continue
        key = str(extraction.value)
        if key not in checked:
            eq = check_equivalent(prediction=extraction.value, gold=gold)
            checked[key] = bool(eq.equivalent)
        if checked[key]:
            return True
    return False


def oracle_action_label(samples: list[Sample], gold: str, probe_k: int = PROBE_K) -> OracleLabelResult:
    """`samples` must be the FULL pool for this problem (all N), sorted
    or not -- this function sorts by `sample_idx` itself so the probe is
    always "the first k generated," matching how a real deployment would
    only have the first k available at decision time.
    """
    ordered = sorted(samples, key=lambda s: s.sample_idx)
    probe = ordered[:probe_k]
    stop_correct = _majority_correct(probe, gold)
    sample_correct = _majority_correct(ordered, gold)
    select_correct = _any_correct(ordered, gold)

    if stop_correct:
        action: OracleAction = "stop"
    elif sample_correct:
        action = "sample"
    elif select_correct:
        action = "select"
    else:
        action = "abstain"
    return OracleLabelResult(
        action=action, stop_correct=stop_correct, sample_correct=sample_correct, select_correct=select_correct
    )
