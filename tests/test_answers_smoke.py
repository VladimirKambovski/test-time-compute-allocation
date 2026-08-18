"""
Day 3 smoke tests for src/marginal_token/answers/, written against
hand-constructed synthetic completions -- NOT a substitute for
tests/test_answer_equivalence.py's golden_200 requirement.

That file needs 200 REAL (model_prediction, gold) pairs hand-checked
against actual policy-model output, which requires the SSH/GPU backend
(not available as of 2026-08-20 -- see notes/2026-08-20.md). This file
instead exercises the extraction/equivalence code's logic against known,
constructed edge cases so the module isn't completely unverified in the
meantime. It stays in the suite permanently as unit coverage even once
golden_200 exists.
"""

from unittest.mock import patch

import pytest

from marginal_token.answers.equivalence import check_equivalent
from marginal_token.answers.extraction import extract_answer, find_boxed_spans
from marginal_token.answers.taxonomy import FailureStatus


# --- find_boxed_spans: brace-balancing ---------------------------------


def test_find_boxed_spans_simple():
    assert find_boxed_spans("The answer is \\boxed{42}.") == ["42"]


def test_find_boxed_spans_nested_braces():
    # This is exactly the case a naive non-brace-balanced regex breaks on.
    text = "So $x = \\boxed{\\frac{1}{2}}$ is the answer."
    assert find_boxed_spans(text) == ["\\frac{1}{2}"]


def test_find_boxed_spans_multiple():
    text = "First \\boxed{41}, then reconsidering, \\boxed{43}."
    assert find_boxed_spans(text) == ["41", "43"]


def test_find_boxed_spans_none():
    assert find_boxed_spans("I don't know the answer.") == []


def test_find_boxed_spans_unbalanced_is_excluded():
    # Simulates a completion truncated mid-box.
    text = "The answer is \\boxed{\\frac{1}{2"
    assert find_boxed_spans(text) == []


# --- extract_answer: the cases that matter for the failure taxonomy ----


def test_extract_answer_single_boxed():
    result = extract_answer("Therefore, $\\boxed{42}$.")
    assert result.status == FailureStatus.OK
    assert result.value is not None


def test_extract_answer_no_boxed_no_fallback():
    result = extract_answer("I'm not sure how to solve this.")
    assert result.status == FailureStatus.NO_BOXED_ANSWER
    assert result.value is None


def test_extract_answer_no_boxed_but_truncated():
    result = extract_answer("Let me work through this step by step, first", finish_reason="length")
    assert result.status == FailureStatus.LENGTH_TRUNCATED


def test_extract_answer_plain_statement_fallback():
    # No \boxed{}, but math_verify's own fallback extraction should catch
    # a plain stated answer.
    result = extract_answer("After simplifying, the answer is 1/2.")
    assert result.status == FailureStatus.OK


def test_extract_answer_agreeing_multi_boxed_is_ok():
    # Same value stated twice (e.g. restated at the end) -- must not be
    # flagged ambiguous just because there are two spans.
    result = extract_answer("We get \\boxed{1/2}, so the final answer is \\boxed{0.5}.")
    assert result.status == FailureStatus.OK


def test_extract_answer_disagreeing_multi_boxed_is_ambiguous():
    # The critical regression case: math_verify's own parse() on this
    # exact text returns the SET {41, 43}, not an ambiguity signal --
    # verified empirically Day 3. This must come back as
    # EXTRACTION_AMBIGUOUS, never silently scored against a set.
    result = extract_answer("First I get \\boxed{41}, but wait, let me redo: \\boxed{43}.")
    assert result.status == FailureStatus.EXTRACTION_AMBIGUOUS
    assert result.value is None


def test_extract_answer_empty_boxed_is_no_boxed_answer():
    result = extract_answer("\\boxed{}")
    assert result.status == FailureStatus.NO_BOXED_ANSWER


# --- check_equivalent: correctness and the timeout-vs-False distinction ---


def test_check_equivalent_true_case():
    from math_verify import parse

    pred = parse("\\frac{1}{2}")[0]
    gold = parse("0.5")[0]
    result = check_equivalent(prediction=pred, gold=gold)
    assert result.equivalent is True
    assert result.status == FailureStatus.OK


def test_check_equivalent_false_case():
    from math_verify import parse

    pred = parse("41")[0]
    gold = parse("42")[0]
    result = check_equivalent(prediction=pred, gold=gold)
    assert result.equivalent is False
    assert result.status == FailureStatus.OK  # a confirmed "not equivalent" IS a valid ok-status result


def test_check_equivalent_timeout_is_not_silently_false():
    # The exact bug this module exists to prevent: math_verify's own
    # verify(..., raise_on_error=False) default would return bare False on
    # a timeout, indistinguishable from a genuine mismatch. Simulate that
    # timeout and assert we surface it as an explicit status instead.
    from math_verify.errors import TimeoutException

    with patch("marginal_token.answers.equivalence.verify", side_effect=TimeoutException):
        result = check_equivalent(prediction="anything", gold="anything")

    assert result.equivalent is None  # NOT False
    assert result.status == FailureStatus.EQUIVALENCE_TIMEOUT
