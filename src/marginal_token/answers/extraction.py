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

from marginal_token.answers.equivalence import _raw_verify
from marginal_token.answers.taxonomy import FailureStatus
from marginal_token.answers.thread_safety import safe_parse as parse

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

    Real bug found and fixed during the Day-3 golden-200 hand-check
    (2026-08-20): a truncated completion with no `\\boxed{}` used to fall
    through to the plain-text fallback below FIRST, which can pick up a
    spurious intermediate equation from mid-derivation scratch work (e.g.
    a completion cut off by max_tokens mid-computation, containing "$c =
    1$" as an aside) and silently credit it as the final answer -- on two
    real completions this produced a false "equivalent" verdict against
    an unrelated gold answer. `finish_reason == "length"` with no boxed
    span is now checked FIRST and always wins: a cut-off derivation isn't
    a confirmed final answer no matter what fragments it contains.
    """
    boxed_spans = find_boxed_spans(text)

    if not boxed_spans:
        if finish_reason == "length":
            return ExtractionResult(value=None, raw_text=None, status=FailureStatus.LENGTH_TRUNCATED)
        # Fall back to math_verify's own non-boxed extraction (plain
        # expressions, bare LaTeX) for completions that state an answer
        # without \boxed{} -- e.g. "The answer is 42." Restricted to the
        # trailing slice of the text, not the full completion: our system
        # prompt always asks for a boxed answer, so a *long* completion
        # with no box is more likely a formatting miss near the end than
        # something to be found by scanning the entire derivation (which
        # risks matching the same kind of spurious mid-derivation fragment
        # as the truncation bug above, just without truncation to blame).
        tail = text[-300:]
        fallback = parse(tail, parsing_timeout=parsing_timeout)
        if fallback:
            raw = str(fallback[-1]) if len(fallback) > 1 else None
            return ExtractionResult(value=fallback[0], raw_text=raw, status=FailureStatus.OK)
        return ExtractionResult(value=None, raw_text=None, status=FailureStatus.NO_BOXED_ANSWER)

    parsed_candidates: list[tuple[str, Any]] = []
    for span in boxed_spans:
        # Re-wrap the bare span content back in \boxed{...} before parsing
        # -- do NOT parse it bare. A second real bug, found the same day
        # as the truncation one: math_verify.parse() on bare span content
        # (no \boxed{}, no $...$ anchor) silently MISPARSES LaTeX commands
        # instead of failing loudly. Verified empirically:
        # parse("3\\sqrt{5}") -> [3, '3'] (drops \sqrt{5} entirely, wrong
        # value, not just a missed match); parse("\\{1,2,3\\}") -> [3, '3']
        # (a full set collapsed to its last element); parse("(0,5)") ->
        # [5, '5']; parse("2\\pi") -> [2, '2']; parse("1+274i") -> [1, '1'].
        # This is exactly the silent-wrong-not-loud-failure invariant #6
        # exists to prevent, and it's what produced a real false
        # "equivalent" verdict during the golden-200 hand-check (gold
        # 3*sqrt(5), silently extracted as bare 3, verify() correctly said
        # not-equal for THAT comparison -- the danger is the cases where
        # the truncated value coincidentally does match a different gold).
        # Re-parsing via \boxed{...} routes through math_verify's own
        # dedicated boxed-content handling (see boxed_match_priority in
        # its LatexExtractionConfig), which does not have this bug.
        parsed = parse(f"\\boxed{{{span}}}", parsing_timeout=parsing_timeout)
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
