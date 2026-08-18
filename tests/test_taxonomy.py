"""
The failure taxonomy is closed (see CLAUDE.md). An uncomputable metric
must record null + a valid status, never a silent 0/incorrect. An
unrecognized status must fail loudly.

Doesn't depend on real model generation -- pure logic over the taxonomy
module, unlike test_answer_equivalence.py's golden-200 requirement.
"""

import pytest

from marginal_token.answers.taxonomy import (
    HIGH_FAILURE_RATE_THRESHOLD,
    FailureStatus,
    UnknownFailureStatusError,
    flag_high_failure_problems,
    validate_status,
)

ALL_14_STATUSES = [
    "no_boxed_answer",
    "extraction_ambiguous",
    "equivalence_timeout",
    "length_truncated",
    "step_segmentation_failed",
    "prm_score_missing",
    "logprobs_unavailable",
    "pool_incomplete",
    "oom",
    "search_budget_exhausted",
    "beam_collapsed",
    "model_load_failed",
    "backend_unavailable",
    "rate_limited",
]


def test_all_valid_statuses_accepted():
    # "ok" plus the 14 failure statuses named in CLAUDE.md.
    assert len(ALL_14_STATUSES) == 14
    for status in ["ok", *ALL_14_STATUSES]:
        assert validate_status(status).value == status
    # Also accepts FailureStatus members directly, not just strings.
    assert validate_status(FailureStatus.OK) is FailureStatus.OK


def test_unknown_status_fails_loudly():
    with pytest.raises(UnknownFailureStatusError):
        validate_status("incorrect")  # the exact thing invariant #6/#7 forbids coercing into
    with pytest.raises(UnknownFailureStatusError):
        validate_status("not_a_real_status")
    with pytest.raises(UnknownFailureStatusError):
        validate_status("")


def test_high_failure_rate_problems_are_flagged_not_dropped():
    # Exactly at the threshold (20%) must NOT be flagged -- brief.md says
    # ">20%", strictly greater than.
    at_threshold = ["ok"] * 8 + ["no_boxed_answer"] * 2  # 20% non-ok
    over_threshold = ["ok"] * 7 + ["no_boxed_answer"] * 3  # 30% non-ok
    all_ok = ["ok"] * 10

    flagged = flag_high_failure_problems(
        {
            "p_at_threshold": at_threshold,
            "p_over_threshold": over_threshold,
            "p_all_ok": all_ok,
        }
    )

    assert "p_at_threshold" not in flagged
    assert "p_all_ok" not in flagged
    assert "p_over_threshold" in flagged
    assert flagged["p_over_threshold"] == pytest.approx(0.3)

    # Flagging must never mean the problem's statuses were mutated/dropped --
    # this function only reports; it doesn't own filtering.
    assert HIGH_FAILURE_RATE_THRESHOLD == 0.20


def test_flagging_rejects_unknown_status_loudly_too():
    # A problem's status list going through the same closed-taxonomy
    # validation as everywhere else -- an unrecognized status anywhere in
    # the aggregate pipeline must fail loudly, not be silently skipped.
    with pytest.raises(UnknownFailureStatusError):
        flag_high_failure_problems({"p1": ["ok", "incorrect"]})
