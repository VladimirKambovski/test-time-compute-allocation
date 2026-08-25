"""Equivalence checking between an extracted prediction and a gold answer.

Two critical details, both verified empirically against the installed
math_verify 0.9.0 (Day 3, 2026-08-20), not assumed from docs -- and both
found via the golden-200 hand-check catching a result that was
implausible on its face (196/196 "not equivalent" against real MATH-500/
OlympiadBench completions):

1. `verify(..., raise_on_error=False)` -- the default -- swallows timeouts
   and internal errors and returns plain `False`. That is indistinguishable
   from a genuine "not equivalent" result, which is precisely the
   silent-incorrect failure mode CLAUDE.md invariant #6 exists to prevent
   ("never silently scored as 0/incorrect"). So this module always calls
   `verify(..., raise_on_error=True)` and converts the resulting
   `TimeoutException` (or any other verification error) into an explicit
   `equivalence_timeout` status instead of a bare `False`.

2. `verify()` does NOT reliably canonicalize a raw, un-parsed gold string
   the way it does a `math_verify.parse()`-produced SymPy object -- despite
   its type signature accepting a plain `str` for `gold`. Empirically:
   `verify("4", parse("\\boxed{4}")[0])` -> False. `verify(parse("4")[0],
   parse("\\boxed{4}")[0])` -> True. Both dev benchmarks (MATH-500,
   OlympiadBench) store gold answers as raw strings, so this is not an
   edge case -- it's the normal input shape. `check_equivalent` now always
   parses a string `gold` internally before calling `verify`, rather than
   documenting "raw string accepted" and trusting the type signature.

3. Parsing that gold string is itself not safe to do bare, for the same
   reason extraction.py's boxed-span parsing wasn't: `parse()` on
   unwrapped LaTeX either fails outright (`parse("p - q")` -> `[]`) or,
   worse, silently mis-parses to a WRONG value rather than failing loudly
   (`parse("\\left(3, \\frac{\\pi}{2}\\right)")` -> bare `3`, dropping the
   tuple entirely). The second case produced a real false-negative during
   the golden-200 hand-check: a genuinely correct `(3, pi/2)` prediction
   scored "not equivalent" because gold silently collapsed to bare `3`.
   `_ensure_parsed` always wraps a string value in `$...$` before parsing
   -- verified safe even when the source string is already `$`-wrapped
   (OlympiadBench sometimes stores gold that way already; double-wrapping
   parses identically to single-wrapping).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from math_verify.errors import TimeoutException

from marginal_token.answers.taxonomy import FailureStatus
from marginal_token.answers.thread_safety import safe_parse as parse
from marginal_token.answers.thread_safety import safe_verify as verify


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


def _ensure_parsed(value: Any, parsing_timeout: int = 5) -> tuple[Any, bool]:
    """Return (parsed_value, ok). If `value` is already a non-string
    math_verify/SymPy object, pass it through unchanged. If it's a raw
    string (the normal shape for gold answers straight from a dataset),
    parse it. `ok=False` means parsing produced nothing usable -- distinct
    from "parsed to a value that doesn't match," which is a real answer.
    """
    if not isinstance(value, str):
        return value, True
    # Wrap in $...$ before parsing -- do NOT parse the bare string. Same
    # bug class as extraction.py's boxed-span fix, found the same day:
    # math_verify.parse() on bare/unwrapped LaTeX content either fails
    # outright (parse("p - q") -> []) or, worse, silently mis-parses to a
    # WRONG value (parse("\\left(3, \\frac{\\pi}{2}\\right)") -> bare 3,
    # dropping the tuple structure entirely). The second case produced a
    # real false-negative during the golden-200 hand-check: a genuinely
    # correct (3, pi/2) prediction was scored "not equivalent" because
    # gold silently collapsed to bare 3. Wrapping is safe even if the
    # string is already $-wrapped in the source dataset (verified
    # empirically -- double-wrapping parses identically to single).
    parsed = parse(f"${value}$", parsing_timeout=parsing_timeout)
    if not parsed:
        return None, False
    return parsed[0], True


@dataclass
class EquivalenceResult:
    equivalent: bool | None  # None means "could not be determined" -- see status
    status: FailureStatus


def check_equivalent(prediction: Any, gold: Any, timeout_seconds: int = 5) -> EquivalenceResult:
    """Check whether an extracted `prediction` (already math_verify-parsed,
    e.g. via `extraction.extract_answer(...).value`) matches `gold`.

    `gold` may be a raw string (the normal case -- straight from a
    benchmark dataset's answer column) or an already-parsed value; either
    way it's parsed via math_verify before comparison, never handed to
    `verify()` unparsed (see module docstring, point 2).

    Per math_verify's own docs, `verify` is NOT symmetric: pass the true
    answer as `gold` and the model's answer as `prediction`.
    """
    parsed_gold, gold_ok = _ensure_parsed(gold, parsing_timeout=timeout_seconds)
    if not gold_ok:
        # Gold itself didn't parse -- not a "not equivalent" result, and
        # not the prediction's fault. No dedicated taxonomy status exists
        # for "gold unparseable" (expected to be rare -- benchmark gold
        # answers are curated), so this is treated as an unresolved
        # comparison, same bucket as a timeout, rather than invented as a
        # new status outside the closed taxonomy.
        return EquivalenceResult(equivalent=None, status=FailureStatus.EQUIVALENCE_TIMEOUT)

    result = _raw_verify(parsed_gold, prediction, timeout_seconds=timeout_seconds)
    if result is None:
        return EquivalenceResult(equivalent=None, status=FailureStatus.EQUIVALENCE_TIMEOUT)
    return EquivalenceResult(equivalent=result, status=FailureStatus.OK)
