"""Replays any controller/policy over cached pools at zero inference cost. Powers every offline experiment, ablation, and the demo's benchmark mode."""

from marginal_token.replay.engine import ReplayResult, replay_many, replay_one

__all__ = ["ReplayResult", "replay_many", "replay_one"]
