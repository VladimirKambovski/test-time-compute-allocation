"""Backend-abstracted, config-hashed, resumable candidate generation. This is the only expensive operation in the whole system -- everything downstream reads from cache."""

from marginal_token.generation.sweep import SweepResult, SweepTask, run_sweep

__all__ = ["SweepResult", "SweepTask", "run_sweep"]
