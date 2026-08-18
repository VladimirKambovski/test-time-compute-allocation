"""
Day 3. The single highest-risk piece of the whole project: if this is
wrong, every downstream number is invalid while looking plausible.

TODO (blocked as of 2026-08-20 -- see notes/2026-08-20.md):
- load tests/fixtures/golden_200.json: 200 hand-checked
  (model_prediction, gold_answer, expected_equivalent: bool) triples,
  drawn from both MATH-500 and OlympiadBench
- for each triple, run the actual extraction + canonicalization +
  math_verify equivalence check used in production
- assert predicted equivalence matches expected_equivalent for all 200
- also assert: extraction failures are tagged with the correct failure
  taxonomy status (see CLAUDE.md), never silently treated as incorrect

Blocked on real policy-model completions to hand-check against: this
sandbox has no GPU/CUDA and no working hosted-API path (G0's ≥2-backend
requirement was waived for Qwen3.5-4B, see notes/2026-08-18.md), and the
SSH/GPU machine mentioned for actual generation isn't set up yet. The
`answers/` module itself (extraction.py, equivalence.py, taxonomy.py) is
built and has synthetic-case unit coverage in tests/test_answers_smoke.py
in the meantime -- these two tests specifically need REAL model output,
which synthetic cases can't substitute for (the whole point is catching
failure modes the actual policy model exhibits, which can't be known
without querying it).
"""

import json
from pathlib import Path

import pytest

GOLDEN_200_PATH = Path(__file__).parent / "fixtures" / "golden_200.json"

_skip_reason = (
    "Blocked on real policy-model generation (no GPU/API backend available yet -- "
    "see notes/2026-08-20.md). Remove this skip once tests/fixtures/golden_200.json "
    "is built from real, hand-checked model completions."
)


@pytest.mark.skipif(not GOLDEN_200_PATH.exists(), reason=_skip_reason)
def test_golden_200_pairs():
    from marginal_token.answers.equivalence import check_equivalent
    from marginal_token.answers.extraction import extract_answer

    triples = json.loads(GOLDEN_200_PATH.read_text())
    assert len(triples) == 200

    mismatches = []
    for triple in triples:
        extraction = extract_answer(triple["model_prediction"])
        if extraction.value is None:
            predicted_equivalent = False
        else:
            predicted_equivalent = check_equivalent(
                prediction=extraction.value, gold=triple["gold_answer"]
            ).equivalent
        if predicted_equivalent != triple["expected_equivalent"]:
            mismatches.append(triple)

    assert not mismatches, f"{len(mismatches)}/200 golden pairs mismatched: {mismatches[:5]}"


@pytest.mark.skipif(not GOLDEN_200_PATH.exists(), reason=_skip_reason)
def test_extraction_failure_uses_taxonomy_not_silent_zero():
    from marginal_token.answers.extraction import extract_answer
    from marginal_token.answers.taxonomy import FailureStatus

    triples = json.loads(GOLDEN_200_PATH.read_text())
    for triple in triples:
        if triple.get("expected_extraction_failure"):
            result = extract_answer(triple["model_prediction"])
            assert result.status != FailureStatus.OK
            assert result.value is None
