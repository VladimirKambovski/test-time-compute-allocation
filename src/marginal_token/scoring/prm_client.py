"""
PRM scoring client protocol + the primary (ladder rung 1) hosted
implementation. Promoted from `notes/scratch/day5_prm_client.py`, same
logic, now behind a `PRMClient` Protocol (mirrors `backends.base.Backend`)
so `pipeline.py` never branches on which rung is in use.

Only rung 1 (`HostedQwen25MathPRMClient`, against `prm_primary` in
`configs/backends/hosted-endpoints.yaml`) is implemented -- Gate G3
passed on the first try with this rung (AUROC=0.9934,
`notes/2026-08-23.md`), so rung 2 (`prm_fallback`, Skywork) was never
exercised and is deliberately NOT implemented speculatively here. It has
a genuinely different request/response contract (raw response text +
`step_token` delimiter, server does its own segmentation -- see that
endpoint's entry in `hosted-endpoints.yaml`), not just a different
model, so a real rung-2 client would need its own class against that
contract, built when the PRM ladder's rung-2 fallback actually fires
(docs/brief.md §27.3a), not guessed at now.

Note on docs/brief.md §27.3a: that section's own text says "no hosted
API provides them [PRMs]" and scopes the resource ladder around
local-GPU-only rungs. The mentor's 2026-08-22 endpoint roster overtook
that assumption (see `notes/2026-08-23.md`) -- rung 1 is reachable via a
real hosted `/score` endpoint, no GPU environment needed.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Protocol


@dataclass
class PRMScoreResult:
    num_steps: int
    step_rewards: list[float]
    mean_reward: float
    ok: bool
    error: str | None = None


class PRMClient(Protocol):
    """One interface, multiple ladder rungs -- callers never branch on
    which rung is in use, they call `score()` and check `.ok`.
    """

    def score(self, query: str, steps: list[str]) -> PRMScoreResult: ...


class HostedQwen25MathPRMClient:
    """Ladder rung 1: the primary PRM (Qwen2.5-Math-PRM-7B) via the
    hosted `prm_primary` custom `/score` endpoint (NOT OpenAI-compatible
    chat completions -- a bespoke FastAPI app; see
    `configs/backends/hosted-endpoints.yaml#prm_primary`'s schema).
    Segmentation is the caller's job (`segmentation.py`) -- this client
    scores whatever `steps` list it's given, it does not segment for us.
    """

    ENDPOINT = "https://math-prm.deb11.smoki.mk/score"
    role = "primary_prm"
    rung = 1

    def __init__(self, api_key_env: str = "HOSTED_ENDPOINT_API_KEY", timeout_s: int = 60):
        self.api_key = os.environ[api_key_env]
        self.timeout_s = timeout_s

    def score(self, query: str, steps: list[str]) -> PRMScoreResult:
        if not steps:
            # Never send an empty steps array and let that read as a real
            # score of 0 -- that would silently misrepresent a
            # segmentation failure as a confident PRM judgment
            # (invariant #6/#7). Callers should already be routing an
            # empty `steps` list to STEP_SEGMENTATION_FAILED before ever
            # reaching here (see pipeline.py) -- this is a defensive
            # backstop, not the primary place that's handled.
            return PRMScoreResult(num_steps=0, step_rewards=[], mean_reward=float("nan"), ok=False,
                                    error="empty_steps")
        body = json.dumps({"query": query, "steps": steps}).encode()
        req = urllib.request.Request(
            self.ENDPOINT, data=body,
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


# Static-typing hint only (Protocol structural checks aren't enforced at
# runtime by this line) -- signals intent to type checkers, matching the
# same pattern in backends/hosted_endpoint.py.
_: type[PRMClient] = HostedQwen25MathPRMClient
