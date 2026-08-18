"""The closed failure taxonomy (CLAUDE.md invariant #7 / docs/brief.md §20).

An uncomputable metric records `null` plus one of these statuses -- never a
silent 0/incorrect. Only `answers/` produces `ok`, `no_boxed_answer`,
`extraction_ambiguous`, `equivalence_timeout`, and passes through
`length_truncated` when generation metadata says the completion was cut
off. The remaining statuses are produced by other modules (scoring, pools,
generation, search, backends) as they're built; this module holds the
*shared, closed* enum so every module validates against the same set.

An unrecognized status must fail loudly (`validate_status` raises), never
get silently coerced into one of these.
"""

from __future__ import annotations

from enum import Enum


class FailureStatus(str, Enum):
    OK = "ok"
    NO_BOXED_ANSWER = "no_boxed_answer"
    EXTRACTION_AMBIGUOUS = "extraction_ambiguous"
    EQUIVALENCE_TIMEOUT = "equivalence_timeout"
    LENGTH_TRUNCATED = "length_truncated"
    STEP_SEGMENTATION_FAILED = "step_segmentation_failed"
    PRM_SCORE_MISSING = "prm_score_missing"
    LOGPROBS_UNAVAILABLE = "logprobs_unavailable"
    POOL_INCOMPLETE = "pool_incomplete"
    OOM = "oom"
    SEARCH_BUDGET_EXHAUSTED = "search_budget_exhausted"
    BEAM_COLLAPSED = "beam_collapsed"
    MODEL_LOAD_FAILED = "model_load_failed"
    BACKEND_UNAVAILABLE = "backend_unavailable"
    RATE_LIMITED = "rate_limited"


# Statuses this module (answers/) can itself assign. Every other member of
# FailureStatus is legal in the taxonomy but is some other module's job to
# produce -- answers/ never emits them.
ANSWERS_MODULE_STATUSES = frozenset(
    {
        FailureStatus.OK,
        FailureStatus.NO_BOXED_ANSWER,
        FailureStatus.EXTRACTION_AMBIGUOUS,
        FailureStatus.EQUIVALENCE_TIMEOUT,
        FailureStatus.LENGTH_TRUNCATED,
    }
)

HIGH_FAILURE_RATE_THRESHOLD = 0.20  # docs/brief.md §20: >20% non-ok is flagged, never dropped


class UnknownFailureStatusError(ValueError):
    """Raised when a status string isn't in the closed taxonomy.

    Per invariant #7, an unrecognized status must fail loudly -- it must
    never be silently coerced into `ok` or any other member.
    """


def validate_status(status: str) -> FailureStatus:
    """Validate `status` against the closed taxonomy, or raise.

    Accepts either a `FailureStatus` member or its string value (since
    statuses round-trip through YAML/JSON as plain strings).
    """
    if isinstance(status, FailureStatus):
        return status
    try:
        return FailureStatus(status)
    except ValueError as exc:
        raise UnknownFailureStatusError(
            f"{status!r} is not a recognized failure-taxonomy status. "
            f"Valid statuses: {sorted(s.value for s in FailureStatus)}. "
            "An unrecognized status must fail loudly, not be coerced."
        ) from exc


def flag_high_failure_problems(
    statuses_by_problem: dict[str, list[str]],
) -> dict[str, float]:
    """Return {problem_id: non_ok_fraction} for every problem whose non-`ok`
    fraction exceeds HIGH_FAILURE_RATE_THRESHOLD.

    Per docs/brief.md §20: "Any problem with >20% non-ok samples is flagged
    and reported separately, never dropped." This function only flags --
    callers are responsible for keeping flagged problems in aggregate
    reporting rather than excluding them.
    """
    flagged: dict[str, float] = {}
    for problem_id, statuses in statuses_by_problem.items():
        validated = [validate_status(s) for s in statuses]
        if not validated:
            continue
        non_ok = sum(1 for s in validated if s is not FailureStatus.OK)
        fraction = non_ok / len(validated)
        if fraction > HIGH_FAILURE_RATE_THRESHOLD:
            flagged[problem_id] = fraction
    return flagged
