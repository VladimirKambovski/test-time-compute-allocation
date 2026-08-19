"""
Day 3. The single highest-risk piece of the whole project: if this is
wrong, every downstream number is invalid while looking plausible.

tests/fixtures/golden_200.json: 200 real (model_prediction, gold_answer,
expected_equivalent, expected_extraction_failure, expected_status)
records, drawn from 35 MATH-500 + 30 OlympiadBench-A problems (3 samples
each, +5 MATH-500 top-up to reach exactly 200), generated against the
mentor-hosted Qwen3.5-4B endpoint (configs/backends/hosted-endpoints.yaml)
on 2026-08-20 and hand-checked -- see notes/2026-08-20.md for the full
writeup, including four real bugs this hand-check caught and fixed in
src/marginal_token/answers/ before this fixture was finalized (raw-string
gold parsing, bare LaTeX span mis-parsing, truncation-vs-fallback
ordering, unwrapped gold tuple collapsing). Build script:
notes/scratch/build_golden_200.py + score_golden_200.py.
"""

import json
from pathlib import Path

from marginal_token.answers.equivalence import check_equivalent
from marginal_token.answers.extraction import extract_answer
from marginal_token.answers.taxonomy import FailureStatus

GOLDEN_200_PATH = Path(__file__).parent / "fixtures" / "golden_200.json"


def test_golden_200_pairs():
    triples = json.loads(GOLDEN_200_PATH.read_text())
    assert len(triples) == 200

    mismatches = []
    for triple in triples:
        extraction = extract_answer(triple["model_prediction"], finish_reason=triple.get("finish_reason"))
        if extraction.value is None:
            predicted_equivalent = False
        else:
            predicted_equivalent = check_equivalent(
                prediction=extraction.value, gold=triple["gold_answer"]
            ).equivalent
        if predicted_equivalent != triple["expected_equivalent"]:
            mismatches.append(triple)

    assert not mismatches, f"{len(mismatches)}/200 golden pairs mismatched: {mismatches[:5]}"


def test_extraction_failure_uses_taxonomy_not_silent_zero():
    triples = json.loads(GOLDEN_200_PATH.read_text())
    checked_any = False
    for triple in triples:
        if triple.get("expected_extraction_failure"):
            checked_any = True
            result = extract_answer(triple["model_prediction"], finish_reason=triple.get("finish_reason"))
            assert result.status != FailureStatus.OK
            assert result.value is None
            assert result.status.value == triple["expected_status"]
    assert checked_any  # sanity: the fixture must actually contain failure cases to be a real test
