"""Extract a final answer from a raw model completion.

Layered on top of `math_verify.parse()` rather than a thin passthrough,
because `parse()` alone has a real gotcha we verified empirically (Day 3,
2026-08-20): given two *disagreeing* `\\boxed{}` occurrences, e.g.
"...\\boxed{41}... \\boxed{43}...", `parse()` silently returns the SET
`{41, 43}` rather than flagging anything. Fed straight into `verify()`
against a single gold value, that just reads as "not equivalent" --
exactly the silent-incorrect-instead-of-taxonomy-status failure mode
CLAUDE.md invariant #6/#7 exists to prevent. So this module finds every
`\\boxed{...}` span itself (brace-balanced, so nested LaTeX like
`\\boxed{\\frac{1}{2}}` is handled correctly) and only trusts the result
when every boxed span parses and they're all pairwise equivalent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from math_verify import parse

from marginal_token.answers.equivalence import _raw_verify
from marginal_token.answers.taxonomy import FailureStatus

_BOXED_MARKER = "\\boxed{"


def find_boxed_spans(text: str) -> list[str]:
    """Return the raw inner content of every `\\boxed{...}` in `text`, in
    order of appearance. Brace-balanced, so nested braces are handled.

    An unbalanced trailing `\\boxed{` (never closed -- e.g. the completion
    was cut off mid-box) is not included; it's the caller's job to combine
    this with `finish_reason` to distinguish that from a genuine absence.
    """
    spans: list[str] = []
    start = 0
    while True:
        idx = text.find(_BOXED_MARKER, start)
        if idx == -1:
            break
        content_start = idx + len(_BOXED_MARKER)
        depth = 1
        i = content_start
        while i < len(text) and depth > 0:
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
            i += 1
        if depth != 0:
            break  # unbalanced -- box never closes, stop scanning
        spans.append(text[content_start : i - 1])
        start = i
    return spans


@dataclass
class ExtractionResult:
    value: Any | None  # canonical math_verify/SymPy representation, or None on failure
    raw_text: str | None  # the specific boxed span (or fallback match) the value came from
    status: FailureStatus
    all_boxed_raw: list[str] = field(default_factory=list)  # every boxed span found, for audit


def extract_answer(
    text: str,
    finish_reason: str | None = None,
    parsing_timeout: int = 5,
) -> ExtractionResult:
    """Extract the final answer from a raw completion.

    `finish_reason` is the generation backend's own signal (e.g. "length"
    when vLLM/an API truncated the completion at max_tokens) -- pass it
    through so a genuinely truncated completion with no boxed answer is
    tagged `length_truncated`, not `no_boxed_answer`. `answers/` doesn't
    infer truncation from text content alone; that would be guessing at
    something the generation layer already knows for a fact.
    """
    boxed_spans = find_boxed_spans(text)

    if not boxed_spans:
        # Fall back to math_verify's own non-boxed extraction (plain
        # expressions, bare LaTeX) for completions that state an answer
        # without \boxed{} -- e.g. "The answer is 42."
        fallback = parse(text, parsing_timeout=parsing_timeout)
        if fallback:
            raw = str(fallback[-1]) if len(fallback) > 1 else None
            return ExtractionResult(value=fallback[0], raw_text=raw, status=FailureStatus.OK)
        if finish_reason == "length":
            return ExtractionResult(value=None, raw_text=None, status=FailureStatus.LENGTH_TRUNCATED)
        return ExtractionResult(value=None, raw_text=None, status=FailureStatus.NO_BOXED_ANSWER)

    parsed_candidates: list[tuple[str, Any]] = []
    for span in boxed_spans:
        parsed = parse(span, parsing_timeout=parsing_timeout)
        if parsed:
            parsed_candidates.append((span, parsed[0]))

    if not parsed_candidates:
        # \boxed{} exists syntactically (e.g. \boxed{} empty, or non-math
        # content) but nothing inside it parsed to a usable value. Treated
        # as "no usable boxed answer" rather than "ambiguous" -- ambiguity
        # implies multiple disagreeing *candidates*, and there are none.
        return ExtractionResult(
            value=None, raw_text=boxed_spans[-1], status=FailureStatus.NO_BOXED_ANSWER, all_boxed_raw=boxed_spans
        )

    if len(parsed_candidates) == 1:
        span, value = parsed_candidates[0]
        return ExtractionResult(value=value, raw_text=span, status=FailureStatus.OK, all_boxed_raw=boxed_spans)

    # Multiple boxed spans parsed. Only trust this if they all agree --
    # otherwise a model that second-guesses itself mid-completion (or
    # states an intermediate boxed value before a final one) must not be
    # silently scored against whichever value happens to win math_verify's
    # own internal fallback ordering.
    reference_span, reference_value = parsed_candidates[-1]
    all_agree = True
    for span, value in parsed_candidates[:-1]:
        agrees = _raw_verify(reference_value, value, timeout_seconds=5)
        if agrees is not True:  # False, or None (timeout/error) -- either way, can't confirm agreement
            all_agree = False
            break

    if all_agree:
        return ExtractionResult(
            value=reference_value, raw_text=reference_span, status=FailureStatus.OK, all_boxed_raw=boxed_spans
        )
    return ExtractionResult(
        value=None, raw_text=reference_span, status=FailureStatus.EXTRACTION_AMBIGUOUS, all_boxed_raw=boxed_spans
    )
