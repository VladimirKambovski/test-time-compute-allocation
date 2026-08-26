"""
Day 5: hosted primary-PRM (Qwen2.5-Math-PRM-7B) client, against
configs/backends/hosted-endpoints.yaml#prm_primary's custom /score
endpoint (NOT OpenAI-compatible chat completions -- a bespoke FastAPI
app, see that file's request/response schema).

Note on docs/brief.md §27.3a: that section assumed "no hosted API
provides them" for PRMs and scoped the resource ladder around local/
GPU-only rungs. The mentor's 2026-08-22 endpoint roster overtook that
assumption -- rung 1 (primary 7B PRM) is reachable via a hosted custom
endpoint, not local-GPU-only. Recorded here rather than silently treated
as if the brief's local-GPU framing still applied unmodified.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass

ENDPOINT = "https://math-prm.deb11.smoki.mk/score"


@dataclass
class PRMScoreResult:
    num_steps: int
    step_rewards: list[float]
    mean_reward: float
    ok: bool
    error: str | None = None


class HostedPRMClient:
    """One instance = the primary PRM at ladder rung 1. Segmentation is
    the caller's job (day5_segmentation.py) -- prm_primary scores
    whatever `steps` list it's given, it does not segment for us (per
    hosted-endpoints.yaml's own note).
    """

    def __init__(self, api_key_env: str = "HOSTED_ENDPOINT_API_KEY", timeout_s: int = 60):
        self.api_key = os.environ[api_key_env]
        self.timeout_s = timeout_s

    def score(self, query: str, steps: list[str]) -> PRMScoreResult:
        if not steps:
            # Never send an empty steps array and let that read as a real
            # score of 0 -- that would silently misrepresent a
            # segmentation failure as a confident PRM judgment (invariant #6/#7).
            return PRMScoreResult(num_steps=0, step_rewards=[], mean_reward=float("nan"), ok=False,
                                   error="empty_steps")
        body = json.dumps({"query": query, "steps": steps}).encode()
        req = urllib.request.Request(
            ENDPOINT, data=body,
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                data = json.load(resp)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, KeyError) as exc:
            return PRMScoreResult(num_steps=0, step_rewards=[], mean_reward=float("nan"), ok=False, error=repr(exc))
        return PRMScoreResult(
            num_steps=data["num_steps"],
            step_rewards=data["step_rewards"],
            mean_reward=data["mean_reward"],
            ok=True,
        )
