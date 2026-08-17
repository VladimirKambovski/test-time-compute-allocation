"""
Day 3. The single highest-risk piece of the whole project: if this is
wrong, every downstream number is invalid while looking plausible.

TODO:
- load tests/fixtures/golden_200.json: 200 hand-checked
  (model_prediction, gold_answer, expected_equivalent: bool) triples,
  drawn from both MATH-500 and OlympiadBench
- for each triple, run the actual extraction + canonicalization +
  math_verify equivalence check used in production
- assert predicted equivalence matches expected_equivalent for all 200
- also assert: extraction failures are tagged with the correct failure
  taxonomy status (see CLAUDE.md), never silently treated as incorrect
"""


def test_golden_200_pairs():
    raise NotImplementedError


def test_extraction_failure_uses_taxonomy_not_silent_zero():
    raise NotImplementedError
