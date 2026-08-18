"""Equivalence checking between an extracted prediction and a gold answer.

Critical detail, verified against math_verify 0.9.0's actual source (Day
3, 2026-08-20), not assumed from docs: `verify(..., raise_on_error=False)`
-- the default -- swallows timeouts and internal errors and returns plain
`False`. That is indistinguishable from a genuine "not equivalent" result,
which is precisely the silent-incorrect failure mode CLAUDE.md invariant
#6 exists to prevent ("never silently scored as 0/incorrect"). So this
module always calls `verify(..., raise_on_error=True)` and converts the
resulting `TimeoutException` (or any other verification error) into an
explicit `equivalence_timeout` status instead of a bare `False`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from math_verify import verify
from math_verify.errors import TimeoutException

from marginal_token.answers.taxonomy import FailureStatus


def _raw_verify(gold: Any, target: Any, timeout_seconds: int = 5) -> bool | None:
    """`verify()` with errors surfaced, not swallowed.

    Returns True/False on a real result, or None if verification itself
    failed (timeout or internal error) -- None is the signal callers must
    map to `equivalence_timeout`, never to False.
    """
    try:
        return verify(gold, target, timeout_seconds=timeout_seconds, raise_on_error=True)
    except TimeoutException:
        return None
    except Exception:
        # Any other math_verify internal error (malformed SymPy object,
        # etc.) -- also not a real "not equivalent" result. Re-raising
        # here would fail loudly on a case math_verify itself documents as
        # possible; treating it as an unresolved comparison (timeout-like)
        # is the safer default given invariant #6's "never silent 0."
        return None


@dataclass
class EquivalenceResult:
    equivalent: bool | None  # None means "could not be determined" -- see status
    status: FailureStatus


def check_equivalent(prediction: Any, gold: Any, timeout_seconds: int = 5) -> EquivalenceResult:
    """Check whether an extracted `prediction` (already math_verify-parsed,
    e.g. via `extraction.extract_answer(...).value`) matches `gold`
    (similarly pre-parsed, or a raw string -- `verify` accepts both).

    Per math_verify's own docs, `verify` is NOT symmetric: pass the true
    answer as `gold` and the model's answer as `prediction`.
    """
    result = _raw_verify(gold, prediction, timeout_seconds=timeout_seconds)
    if result is None:
        return EquivalenceResult(equivalent=None, status=FailureStatus.EQUIVALENCE_TIMEOUT)
    return EquivalenceResult(equivalent=result, status=FailureStatus.OK)
