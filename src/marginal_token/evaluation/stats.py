"""Paired bootstrap (10k resamples, BCa), McNemar, Holm-Bonferroni,
difficulty banding. Per docs/brief.md §19:
- Every comparison is a per-problem paired difference.
- 10,000 resamples over the problem axis, BCa intervals, for every
  reported difference.
- McNemar for paired binary correctness flips between actions at fixed
  budget.
- Holm-Bonferroni within declared families (actions; comparators;
  ablations).
- No mean/p-value/CI for any cell with fewer than 5 replicates
  (honesty rule) -- report medians/ordinal statements instead.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MIN_REPLICATES_FOR_INFERENTIAL_STATS = 5  # brief.md §19 "honesty rule"


class InsufficientReplicatesError(ValueError):
    """Raised instead of silently computing a mean/CI/p-value on too few
    replicates -- per the honesty rule, those cells report medians and
    ordinal statements instead, never a point estimate dressed up with
    false precision.
    """


@dataclass
class BootstrapResult:
    point_estimate: float
    ci_lo: float
    ci_hi: float
    n_resamples: int


def paired_bootstrap_bca(
    differences: np.ndarray, n_resamples: int = 10_000, alpha: float = 0.05, seed: int | None = None,
) -> BootstrapResult:
    """BCa (bias-corrected and accelerated) confidence interval on the
    mean of per-problem paired differences. Percentile bootstrap alone
    under- or over-covers when the sampling distribution is skewed;
    BCa corrects for both median bias and skewness via the jackknife.

    `differences` must already be paired -- one value per problem, e.g.
    (oracle_correct - best_fixed_correct) per problem, not two separate
    unpaired arrays.
    """
    n = len(differences)
    if n < MIN_REPLICATES_FOR_INFERENTIAL_STATS:
        raise InsufficientReplicatesError(
            f"Only {n} replicates (<{MIN_REPLICATES_FOR_INFERENTIAL_STATS}) -- report the median "
            f"and an ordinal statement instead, per docs/brief.md §19's honesty rule."
        )

    rng = np.random.default_rng(seed)
    theta_hat = differences.mean()

    boot_idx = rng.integers(0, n, size=(n_resamples, n))
    boot_means = differences[boot_idx].mean(axis=1)

    # Bias-correction factor z0: how far the bootstrap distribution's
    # median is from theta_hat, in normal-quantile units.
    from scipy import stats as _stats  # local import: heavy dep, only needed here

    prop_less = np.mean(boot_means < theta_hat)
    # Guard the boundary cases (all resamples above/below theta_hat)
    prop_less = np.clip(prop_less, 1 / (n_resamples + 1), n_resamples / (n_resamples + 1))
    z0 = _stats.norm.ppf(prop_less)

    # Acceleration factor a, via the jackknife (leave-one-out) skewness.
    jackknife_means = np.array([
        np.delete(differences, i).mean() for i in range(n)
    ])
    jack_mean = jackknife_means.mean()
    numerator = np.sum((jack_mean - jackknife_means) ** 3)
    denominator = 6.0 * (np.sum((jack_mean - jackknife_means) ** 2) ** 1.5)
    a = numerator / denominator if denominator != 0 else 0.0

    z_alpha_lo = _stats.norm.ppf(alpha / 2)
    z_alpha_hi = _stats.norm.ppf(1 - alpha / 2)

    def bca_percentile(z_alpha: float) -> float:
        adjusted = z0 + (z0 + z_alpha) / (1 - a * (z0 + z_alpha))
        return float(_stats.norm.cdf(adjusted)) * 100

    lo_pct = bca_percentile(z_alpha_lo)
    hi_pct = bca_percentile(z_alpha_hi)
    lo_pct, hi_pct = np.clip([lo_pct, hi_pct], 0, 100)

    ci_lo, ci_hi = np.percentile(boot_means, [lo_pct, hi_pct])
    return BootstrapResult(point_estimate=float(theta_hat), ci_lo=float(ci_lo), ci_hi=float(ci_hi),
                            n_resamples=n_resamples)


@dataclass
class McNemarResult:
    statistic: float
    p_value: float
    n_discordant: int


def mcnemar_test(a_correct: np.ndarray, b_correct: np.ndarray, continuity_correction: bool = True) -> McNemarResult:
    """Paired binary test for correctness flips between two actions at a
    fixed budget, on the SAME problems. Only the discordant pairs (one
    right, one wrong) carry information.
    """
    a_correct = np.asarray(a_correct, dtype=bool)
    b_correct = np.asarray(b_correct, dtype=bool)
    if len(a_correct) != len(b_correct):
        raise ValueError("a_correct and b_correct must be paired (same length, same problem order)")

    n01 = int(np.sum(~a_correct & b_correct))  # a wrong, b right
    n10 = int(np.sum(a_correct & ~b_correct))  # a right, b wrong
    n_discordant = n01 + n10

    if n_discordant < MIN_REPLICATES_FOR_INFERENTIAL_STATS:
        raise InsufficientReplicatesError(
            f"Only {n_discordant} discordant pairs (<{MIN_REPLICATES_FOR_INFERENTIAL_STATS}) -- "
            f"report medians/ordinal statement instead, per the honesty rule."
        )

    from scipy import stats as _stats

    if continuity_correction:
        statistic = (abs(n01 - n10) - 1) ** 2 / (n01 + n10)
    else:
        statistic = (n01 - n10) ** 2 / (n01 + n10)
    p_value = float(1 - _stats.chi2.cdf(statistic, df=1))
    return McNemarResult(statistic=float(statistic), p_value=p_value, n_discordant=n_discordant)


def holm_bonferroni(p_values: dict[str, float], alpha: float = 0.05) -> dict[str, bool]:
    """Sequential Holm-Bonferroni correction within one declared family
    (e.g. all pairwise action comparisons, or all comparators, or all
    ablations -- per §19, corrected WITHIN a family, not globally across
    unrelated families).

    Returns {label: reject_null} for each input p-value.
    """
    if not p_values:
        return {}
    labels_sorted = sorted(p_values, key=lambda k: p_values[k])
    m = len(labels_sorted)
    reject = {}
    still_rejecting = True
    for i, label in enumerate(labels_sorted):
        threshold = alpha / (m - i)
        if still_rejecting and p_values[label] <= threshold:
            reject[label] = True
        else:
            still_rejecting = False  # Holm's step-down: once one fails, all subsequent fail too
            reject[label] = False
    return reject


def difficulty_bands(pass_at_1: dict[str, float], n_bands: int = 5) -> dict[str, int]:
    """Assign each problem to one of `n_bands` difficulty bands by its
    pass@1 rate, per docs/brief.md §19 ("headline results reported
    overall and by five pass@1 bands"). Band 0 = hardest (lowest pass@1),
    band n_bands-1 = easiest. Ties broken by stable sort order (problem
    id), not randomly, so banding is reproducible.
    """
    ordered = sorted(pass_at_1.items(), key=lambda kv: (kv[1], kv[0]))
    n = len(ordered)
    bands: dict[str, int] = {}
    for i, (pid, _) in enumerate(ordered):
        band = min(n_bands - 1, (i * n_bands) // n)
        bands[pid] = band
    return bands
