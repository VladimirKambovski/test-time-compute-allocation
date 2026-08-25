"""Offline batch PRM scoring (separate process, separate model load from generation) and an online single-sample scoring path for the live gateway."""

from marginal_token.scoring.pipeline import (
    PRMScore,
    PRMScoreStore,
    compute_score_id,
    score_pool,
    score_sample,
)
from marginal_token.scoring.prm_client import HostedQwen25MathPRMClient, PRMClient, PRMScoreResult
from marginal_token.scoring.segmentation import DEFAULT_CONVENTION, segment

__all__ = [
    "PRMScore",
    "PRMScoreStore",
    "compute_score_id",
    "score_pool",
    "score_sample",
    "HostedQwen25MathPRMClient",
    "PRMClient",
    "PRMScoreResult",
    "DEFAULT_CONVENTION",
    "segment",
]
