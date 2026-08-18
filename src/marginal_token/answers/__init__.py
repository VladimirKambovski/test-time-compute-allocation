"""Answer extraction, canonicalization, and equivalence checking via math_verify.

Highest-risk module in the repo -- see tests/test_answer_equivalence.py.
"canonicalization" is handled by math_verify's own `parse()` (it returns a
canonical SymPy representation); this package's job is orchestrating
extraction (finding and disambiguating candidate answers in raw text) and
equivalence (checking a canonicalized prediction against gold) so that
every failure mode maps to an explicit taxonomy status rather than a
silent incorrect/0.
"""

from marginal_token.answers.equivalence import EquivalenceResult, check_equivalent
from marginal_token.answers.extraction import ExtractionResult, extract_answer, find_boxed_spans
from marginal_token.answers.taxonomy import (
    ANSWERS_MODULE_STATUSES,
    HIGH_FAILURE_RATE_THRESHOLD,
    FailureStatus,
    UnknownFailureStatusError,
    flag_high_failure_problems,
    validate_status,
)

__all__ = [
    "ANSWERS_MODULE_STATUSES",
    "HIGH_FAILURE_RATE_THRESHOLD",
    "EquivalenceResult",
    "ExtractionResult",
    "FailureStatus",
    "UnknownFailureStatusError",
    "check_equivalent",
    "extract_answer",
    "find_boxed_spans",
    "flag_high_failure_problems",
    "validate_status",
]
