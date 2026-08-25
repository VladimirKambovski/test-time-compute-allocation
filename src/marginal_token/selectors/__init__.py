"""Selectors turn a pool of samples into one answer: plain majority, oracle pass@k, PRM-weighted majority, PRM-argmax (MUST); self-certainty, length-normalized, cluster-then-vote, PRM reductions (SHOULD)."""

from marginal_token.selectors.basic import (
    MajorityResult,
    VoteEntry,
    accuracy,
    oracle_pass_at_k,
    pass_at_k_unbiased,
    plain_majority,
    votes_from_samples,
)
from marginal_token.selectors.prm_based import WeightedVoteEntry, prm_argmax, prm_weighted_majority

__all__ = [
    "MajorityResult",
    "VoteEntry",
    "accuracy",
    "oracle_pass_at_k",
    "pass_at_k_unbiased",
    "plain_majority",
    "votes_from_samples",
    "WeightedVoteEntry",
    "prm_argmax",
    "prm_weighted_majority",
]
