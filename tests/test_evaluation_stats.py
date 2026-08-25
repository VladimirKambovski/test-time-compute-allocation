"""
Day 8's explicit "done when": bootstrap reproduces a known CI on
synthetic data. `evaluation/stats.py` was built on the Day-4 safe list
but had never been tested until now -- this is that first real test
pass, not a re-verification of something already checked.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats as scipy_stats

from marginal_token.evaluation.stats import (
    MIN_REPLICATES_FOR_INFERENTIAL_STATS,
    InsufficientReplicatesError,
    difficulty_bands,
    holm_bonferroni,
    mcnemar_test,
    paired_bootstrap_bca,
)

# --- paired_bootstrap_bca -----------------------------------------------


def test_bootstrap_collapses_to_the_exact_value_on_zero_variance_data():
    """The one truly *known*, non-approximate CI: if every paired
    difference is identical, every bootstrap resample's mean is that
    same constant, so the CI must collapse to exactly that value with
    zero width -- a strong check that the bias-correction/acceleration
    machinery doesn't produce NaN or garbage in a degenerate case.
    """
    differences = np.full(20, 0.1)
    result = paired_bootstrap_bca(differences, n_resamples=2000, seed=0)
    assert result.point_estimate == pytest.approx(0.1)
    assert result.ci_lo == pytest.approx(0.1)
    assert result.ci_hi == pytest.approx(0.1)


def test_bootstrap_ci_matches_the_normal_approximation_on_well_behaved_data():
    """A known population (fixed seed, fixed mean/std): BCa on
    large-n, near-symmetric data should closely track the textbook
    normal-approximation CI (mean +/- 1.96*sem). This is the standard
    way to validate a bootstrap CI implementation against a known
    answer when the data isn't degenerate.
    """
    rng = np.random.default_rng(42)
    true_mean, true_std, n = 0.08, 0.10, 500
    differences = rng.normal(true_mean, true_std, size=n)

    result = paired_bootstrap_bca(differences, n_resamples=10_000, seed=123)

    sample_mean = differences.mean()
    sem = differences.std(ddof=1) / np.sqrt(n)
    normal_lo, normal_hi = sample_mean - 1.96 * sem, sample_mean + 1.96 * sem

    assert result.point_estimate == pytest.approx(sample_mean)
    # Generous tolerance -- BCa and the normal approximation agree
    # closely but not exactly on near-symmetric data; this checks
    # "reproduces the known CI," not "is bit-identical to it."
    assert result.ci_lo == pytest.approx(normal_lo, abs=0.02)
    assert result.ci_hi == pytest.approx(normal_hi, abs=0.02)


def test_bootstrap_raises_below_the_minimum_replicate_floor():
    differences = np.array([0.1, 0.2, 0.0, -0.1])  # 4 < MIN_REPLICATES_FOR_INFERENTIAL_STATS
    assert len(differences) < MIN_REPLICATES_FOR_INFERENTIAL_STATS
    with pytest.raises(InsufficientReplicatesError):
        paired_bootstrap_bca(differences)


# --- mcnemar_test --------------------------------------------------------


def test_mcnemar_statistic_matches_the_closed_form_by_hand():
    # 10 problems where a is wrong/b is right, 2 where a is right/b is
    # wrong -- n_discordant = 12, well over the honesty-rule floor.
    a_correct = np.array([False] * 10 + [True] * 2 + [True] * 20)
    b_correct = np.array([True] * 10 + [False] * 2 + [True] * 20)
    result = mcnemar_test(a_correct, b_correct, continuity_correction=True)
    expected_statistic = (abs(10 - 2) - 1) ** 2 / (10 + 2)  # = 49/12
    assert result.statistic == pytest.approx(expected_statistic)
    assert result.n_discordant == 12
    # cross-check the p-value against scipy's own chi2 survival function
    # directly (not re-deriving mcnemar_test's formula, just confirming
    # it actually plugged the statistic into the right distribution).
    assert result.p_value == pytest.approx(1 - scipy_stats.chi2.cdf(expected_statistic, df=1))
    assert result.p_value < 0.05  # 4.083 > 3.841, the df=1 critical value at alpha=0.05


def test_mcnemar_raises_below_the_minimum_discordant_floor():
    a_correct = np.array([True, True, False, True, True])
    b_correct = np.array([True, True, True, True, True])  # only 1 discordant pair
    with pytest.raises(InsufficientReplicatesError):
        mcnemar_test(a_correct, b_correct)


def test_mcnemar_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        mcnemar_test(np.array([True, False]), np.array([True]))


# --- holm_bonferroni ------------------------------------------------------


def test_holm_bonferroni_matches_a_hand_worked_example():
    # alpha=0.05, m=4. Sorted ascending: a(.001), b(.01), c(.03), d(.5).
    # Thresholds: a<=.05/4=.0125 reject; b<=.05/3=.01667 reject;
    # c<=.05/2=.025 -- .03 > .025, fail, step-down stops; d also fails.
    p_values = {"a": 0.001, "b": 0.01, "c": 0.03, "d": 0.5}
    result = holm_bonferroni(p_values, alpha=0.05)
    assert result == {"a": True, "b": True, "c": False, "d": False}


def test_holm_bonferroni_empty_input_returns_empty():
    assert holm_bonferroni({}) == {}


# --- difficulty_bands ------------------------------------------------------


def test_difficulty_bands_matches_a_hand_worked_assignment():
    pass_at_1 = {f"p{i}": i / 10 for i in range(10)}  # p0=0.0 (hardest) ... p9=0.9 (easiest)
    bands = difficulty_bands(pass_at_1, n_bands=5)
    expected = {"p0": 0, "p1": 0, "p2": 1, "p3": 1, "p4": 2, "p5": 2, "p6": 3, "p7": 3, "p8": 4, "p9": 4}
    assert bands == expected


def test_difficulty_bands_ties_broken_by_problem_id_not_randomly():
    pass_at_1 = {"z": 0.5, "a": 0.5}  # identical pass@1 -- order must be deterministic
    bands_first = difficulty_bands(pass_at_1, n_bands=2)
    bands_second = difficulty_bands(pass_at_1, n_bands=2)
    assert bands_first == bands_second
    assert bands_first["a"] < bands_first["z"]  # "a" sorts first on the tie-break
