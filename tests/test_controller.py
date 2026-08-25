"""
Day 10: `featurize()`, `oracle_action_label()`, and the 7 fixed
policies. Synthetic data throughout except where noted -- the real
cross-check against P1 (500 real problems) is
`notes/scratch/day10_fit_predictors.py`, run separately and logged in
`notes/2026-08-23.md`, not repeated here as a slow test.
"""

from __future__ import annotations

import math

import pytest

from marginal_token.backends.base import Provenance, Sample, now_iso
from marginal_token.controller.base import Budget, Probe
from marginal_token.controller.features import FEATURE_NAMES, featurize
from marginal_token.controller.oracle_labels import oracle_action_label
from marginal_token.controller.policies import (
    GamblerController,
    MiserController,
    OracleController,
    SpendthriftController,
    UniformSelectController,
)
from marginal_token.controller.predictor import DetectiveController, FortuneTellerController


def _sample(text: str, sample_idx: int, finish_reason: str = "stop", logprobs=None) -> Sample:
    provenance = Provenance(
        model_id="fake", backend="api_host", provider="fake", revision_or_api_model="fake",
        temperature=0.8, top_p=0.95, max_tokens=1024, seed=None, seed_honored="unknown",
        logprobs_available=logprobs is not None, quantization="none", generated_at=now_iso(), pool_id="p",
    )
    return Sample(text=text, finish_reason=finish_reason, completion_tokens=len(text.split()),
                   prompt_tokens=10, provenance=provenance, problem_id="p0", sample_idx=sample_idx,
                   logprobs=logprobs)


# --- featurize() ---------------------------------------------------------


def test_featurize_unanimous_probe_has_max_agreement_zero_entropy():
    samples = [_sample(f"\\boxed{{4}}\n\nstep {i}", i) for i in range(4)]
    feats = featurize(Probe(samples=samples))
    assert feats["top1_vote_fraction"] == 1.0
    assert feats["top2_margin"] == 1.0
    assert feats["normalized_entropy"] == 0.0
    assert feats["distinct_answer_count"] == 1.0


def test_featurize_split_probe_has_partial_agreement_and_positive_entropy():
    samples = [
        _sample("\\boxed{4}\n\nstep", 0), _sample("\\boxed{4}\n\nstep", 1),
        _sample("\\boxed{5}\n\nstep", 2), _sample("\\boxed{6}\n\nstep", 3),
    ]
    feats = featurize(Probe(samples=samples))
    assert feats["top1_vote_fraction"] == 0.5  # 2/4 voted "4"
    assert feats["top2_margin"] == pytest.approx(0.25)  # (2-1)/4
    assert feats["distinct_answer_count"] == 3.0
    assert feats["normalized_entropy"] > 0.0


def test_featurize_extraction_failures_lower_vote_fraction_not_hidden():
    samples = [
        _sample("\\boxed{4}\n\nstep", 0),
        _sample("no boxed answer at all here", 1, finish_reason="length"),
        _sample("no boxed answer at all here", 2, finish_reason="length"),
        _sample("no boxed answer at all here", 3, finish_reason="length"),
    ]
    feats = featurize(Probe(samples=samples))
    assert feats["top1_vote_fraction"] == 0.25  # 1/4, not 1/1 -- denominator is ALL probe samples
    assert feats["extraction_failure_fraction"] == 0.75
    assert feats["truncation_fraction"] == 0.75


def test_featurize_confidence_group_is_nan_without_logprobs():
    samples = [_sample("\\boxed{4}\n\nstep", i) for i in range(4)]  # no logprobs passed
    feats = featurize(Probe(samples=samples))
    assert math.isnan(feats["mean_logprob"])
    assert math.isnan(feats["min_logprob"])
    assert math.isnan(feats["self_certainty"])
    assert math.isnan(feats["cumulative_logprob_spread"])


def test_featurize_confidence_group_is_populated_with_logprobs():
    lp = [{"logprob": -0.1, "top_logprobs": [{"logprob": -0.1}, {"logprob": -2.0}]} for _ in range(5)]
    samples = [_sample("\\boxed{4}\n\nstep", i, logprobs=lp) for i in range(4)]
    feats = featurize(Probe(samples=samples))
    assert feats["mean_logprob"] == pytest.approx(-0.1)
    assert feats["min_logprob"] == pytest.approx(-0.1)
    assert not math.isnan(feats["self_certainty"])


def test_featurize_output_matches_declared_feature_names_exactly():
    samples = [_sample("\\boxed{4}\n\nstep", 0)]
    feats = featurize(Probe(samples=samples))
    assert set(feats) == set(FEATURE_NAMES)


def test_featurize_rejects_empty_probe():
    with pytest.raises(ValueError):
        featurize(Probe(samples=[]))


# --- oracle_action_label() -----------------------------------------------


def test_oracle_label_stop_when_probe_already_correct():
    # All 32 "generated" samples agree and are correct -- probe (first 4) is too.
    samples = [_sample("\\boxed{4}", i) for i in range(32)]
    result = oracle_action_label(samples, gold="4")
    assert result.action == "stop"
    assert result.stop_correct is True
    assert result.sample_correct is True


def test_oracle_label_sample_when_probe_wrong_but_full_pool_right():
    # First 4 (the probe) all say "5" (wrong); samples 4-31 all say "4" (right, and the majority overall).
    samples = [_sample("\\boxed{5}", i) for i in range(4)] + [_sample("\\boxed{4}", i) for i in range(4, 32)]
    result = oracle_action_label(samples, gold="4")
    assert result.stop_correct is False
    assert result.sample_correct is True
    assert result.action == "sample"


def test_oracle_label_select_when_majority_wrong_but_a_lone_sample_is_right():
    # Probe (first 4) all wrong; full-pool majority ("5", 27 votes) is
    # also wrong -- but ONE sample (idx 31) says "4" and is right. A
    # perfect PRM-based selector could have found it even though plain
    # majority never would -- that's exactly what SELECT should catch.
    samples = [_sample("\\boxed{5}", i) for i in range(31)] + [_sample("\\boxed{4}", 31)]
    result = oracle_action_label(samples, gold="4")
    assert result.stop_correct is False
    assert result.sample_correct is False
    assert result.select_correct is True
    assert result.action == "select"


def test_oracle_label_abstain_when_neither_probe_nor_full_pool_is_right():
    samples = [_sample("\\boxed{5}", i) for i in range(32)]
    result = oracle_action_label(samples, gold="4")
    assert result.action == "abstain"
    assert result.stop_correct is False
    assert result.sample_correct is False
    assert result.select_correct is False


def test_oracle_label_uses_the_first_k_generated_as_the_probe_regardless_of_input_order():
    # Shuffle the list -- oracle_action_label must sort by sample_idx
    # itself, since a real deployment only ever has "the first k
    # generated" available at decision time, not whatever order a list
    # happens to arrive in.
    ordered_wrong_first = [_sample("\\boxed{5}", i) for i in range(4)] + \
                            [_sample("\\boxed{4}", i) for i in range(4, 32)]
    shuffled = list(reversed(ordered_wrong_first))
    result = oracle_action_label(shuffled, gold="4")
    assert result.action == "sample"  # same result as the pre-sorted version


# --- fixed policies -------------------------------------------------------


def _probe_and_budget():
    return Probe(samples=[_sample("\\boxed{4}", 0)]), Budget(max_tokens=1024)


def test_miser_always_stops_zero_grant():
    probe, budget = _probe_and_budget()
    decision = MiserController().decide(probe, budget)
    assert decision.action == "stop"
    assert decision.budget_grant == 0


def test_spendthrift_always_samples_full_grant():
    probe, budget = _probe_and_budget()
    decision = SpendthriftController().decide(probe, budget)
    assert decision.action == "sample"
    assert decision.budget_grant == budget.max_tokens


def test_uniform_select_always_selects():
    probe, budget = _probe_and_budget()
    decision = UniformSelectController().decide(probe, budget)
    assert decision.action == "select"


def test_gambler_is_deterministic_under_a_fixed_seed():
    probe, budget = _probe_and_budget()
    c1 = GamblerController(stop_probability=0.3, seed=42)
    c2 = GamblerController(stop_probability=0.3, seed=42)
    actions1 = [c1.decide(probe, budget).action for _ in range(20)]
    actions2 = [c2.decide(probe, budget).action for _ in range(20)]
    assert actions1 == actions2


def test_gambler_rejects_invalid_probability():
    with pytest.raises(ValueError):
        GamblerController(stop_probability=1.5)


def test_oracle_controller_requires_precomputed_label():
    probe, budget = _probe_and_budget()
    with pytest.raises(ValueError):
        OracleController().decide(probe, budget)


def test_oracle_controller_uses_the_precomputed_label():
    budget = Budget(max_tokens=1024)
    probe = Probe(samples=[], features={"oracle_action": "sample"})
    decision = OracleController().decide(probe, budget)
    assert decision.action == "sample"
    assert decision.budget_grant == budget.max_tokens


# --- DetectiveController (scaffold) ---------------------------------------


def test_detective_fits_and_predicts_on_separable_synthetic_features():
    # Two obviously-separable classes: high agreement -> "stop", low
    # agreement -> "sample". A working logistic fit should recover this.
    X = [
        {**{f: 0.0 for f in FEATURE_NAMES}, "top1_vote_fraction": 1.0, "normalized_entropy": 0.0},
        {**{f: 0.0 for f in FEATURE_NAMES}, "top1_vote_fraction": 0.95, "normalized_entropy": 0.05},
        {**{f: 0.0 for f in FEATURE_NAMES}, "top1_vote_fraction": 0.25, "normalized_entropy": 0.9},
        {**{f: 0.0 for f in FEATURE_NAMES}, "top1_vote_fraction": 0.3, "normalized_entropy": 0.85},
    ] * 5
    y = ["stop", "stop", "sample", "sample"] * 5
    controller = DetectiveController()
    controller.fit(X, y)

    high_agreement_probe = Probe(samples=[_sample("\\boxed{4}", i) for i in range(4)])
    decision = controller.decide(high_agreement_probe, Budget(max_tokens=1024))
    assert decision.action == "stop"


def test_detective_raises_if_asked_to_decide_before_fit():
    probe, budget = _probe_and_budget()
    with pytest.raises(RuntimeError):
        DetectiveController().decide(probe, budget)


def test_detective_imputes_nan_confidence_columns_rather_than_crashing():
    # All-NaN confidence columns (no logprobs anywhere in training data,
    # exactly P1's real situation) must not crash the fit.
    X = [{**{f: 0.0 for f in FEATURE_NAMES}, "mean_logprob": float("nan")}] * 4
    y = ["stop", "sample", "stop", "sample"]
    controller = DetectiveController()
    controller.fit(X, y)  # must not raise
    assert controller._impute_values["mean_logprob"] == 0.0  # nothing to impute from -> falls back to 0


# --- FortuneTellerController (scaffold, fake embedder -- no network) -----


class _FakeEmbedder:
    """Deterministic stand-in for SentenceTransformer -- 2D embeddings
    keyed by a keyword in the text, so fit/predict can be checked without
    downloading a real model in every test run.
    """

    def encode(self, texts):
        import numpy as np

        return np.array([[1.0, 0.0] if "easy" in t else [0.0, 1.0] for t in texts])


def test_fortune_teller_fits_and_predicts_from_query_text_alone():
    controller = FortuneTellerController()
    controller._embedder = _FakeEmbedder()  # bypass the real model download for this test
    controller.fit(["easy question", "easy one too", "hard question", "hard one too"] * 3,
                     ["stop", "stop", "sample", "sample"] * 3)

    probe = Probe(samples=[], features={"query_text": "an easy question"})
    decision = controller.decide(probe, Budget(max_tokens=1024))
    assert decision.action == "stop"


def test_fortune_teller_requires_query_text_in_probe_features():
    controller = FortuneTellerController()
    controller._embedder = _FakeEmbedder()
    controller.fit(["easy", "hard"], ["stop", "sample"])
    with pytest.raises(ValueError):
        controller.decide(Probe(samples=[]), Budget(max_tokens=1024))
