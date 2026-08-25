"""The mentor-provided hosted Qwen3.5-4B endpoint (configs/backends/
hosted-endpoints.yaml). See notes/2026-08-20.md for the full verification
transcript this implementation is built from.

IMPORTANT_CAVEAT (carried from hosted-endpoints.yaml, repeated here so it
isn't lost in code review): this endpoint serves a third-party
`unsloth/Qwen3.5-4B-GGUF:BF16` conversion via llama.cpp, not the official
`Qwen/Qwen3.5-4B` safetensors via vLLM pinned in configs/policies/
qwen3.5-4b.yaml. Recorded honestly in `provider`/`revision_or_api_model`
below -- never silently reported as if it were the official checkpoint.
"""

from __future__ import annotations

import json
import os
import urllib.request
from typing import Any

from marginal_token.backends.base import (
    Backend,
    BackendCaps,
    DecodeConfig,
    Provenance,
    Sample,
    now_iso,
)

ENDPOINT = "https://qwen35-4b-bf16.deb12.smoki.mk/v1/chat/completions"
SERVED_MODEL_ID = "unsloth/Qwen3.5-4B-GGUF:BF16"
DEFAULT_SYSTEM_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}."


class HostedQwen35Backend:
    """Implements the `Backend` protocol against the mentor-hosted
    endpoint. One instance = one fixed decode contract; construct a new
    one (and treat it as a new pool condition) if any contract field
    changes -- see Provenance.CONTRACT_FIELDS.
    """

    backend_type = "api_host"
    provider = "mentor_hosted_llamacpp"
    # Honest, not the official repo -- see module docstring.
    revision_or_api_model = SERVED_MODEL_ID
    quantization = "bf16_gguf_third_party_conversion"

    def __init__(
        self, api_key_env: str = "HOSTED_ENDPOINT_API_KEY", pool_id: str = "", timeout_s: int = 300,
        request_logprobs: bool = True, top_logprobs: int = 20,
    ):
        self.api_key = os.environ[api_key_env]
        self.pool_id = pool_id
        self.timeout_s = timeout_s
        # Real bug found 2026-08-23 (Day 10): `capabilities()` below has
        # claimed `logprobs_available=True` since Day 3's endpoint
        # verification, but this class never actually requested them --
        # P1's entire 16,000-sample pool (Day 6-9) was generated with
        # this gap, so it has ZERO logprob data despite the backend
        # genuinely supporting it (verified up to top_logprobs=100,
        # notes/2026-08-20.md). Fixed here for all FUTURE generation
        # (P2, held-out, any N=64 extension) -- NOT retroactive; P1
        # itself would need full regeneration to add logprobs after the
        # fact, which is a real cost/time decision, not something to do
        # silently. docs/brief.md §16's "confidence" feature group is
        # simply unavailable for P1; the predictor degrades gracefully
        # to the agreement/shape/hygiene groups on that pool, per §16's
        # own documented tolerance for exactly this situation.
        self.request_logprobs = request_logprobs
        self.top_logprobs = top_logprobs

    def capabilities(self) -> BackendCaps:
        # Verified 2026-08-20: full per-token logprobs for every generated
        # token (not sampled), top_logprobs honored up to at least 100.
        # seed accepted but bit-reproducibility never confirmed -> "unknown".
        return BackendCaps(logprobs_available=True, seed_honored="unknown", max_concurrency_hint=6)

    def generate(self, prompts: list[str], cfg: DecodeConfig, system_prompt: str = DEFAULT_SYSTEM_PROMPT) -> list[Sample]:
        return [self._generate_one(p, cfg, system_prompt) for p in prompts]

    def _generate_one(self, prompt: str, cfg: DecodeConfig, system_prompt: str) -> Sample:
        body: dict[str, Any] = {
            "model": SERVED_MODEL_ID,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": cfg.max_tokens,
            "temperature": cfg.temperature,
            "top_p": cfg.top_p,
            "chat_template_kwargs": {"enable_thinking": cfg.thinking_mode},
        }
        if cfg.seed is not None:
            body["seed"] = cfg.seed
        if cfg.stop_sequences:
            body["stop"] = list(cfg.stop_sequences)
        if self.request_logprobs:
            body["logprobs"] = True
            body["top_logprobs"] = self.top_logprobs

        req = urllib.request.Request(
            ENDPOINT,
            data=json.dumps(body).encode(),
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
            data = json.load(resp)

        choice = data["choices"][0]
        usage = data.get("usage", {})
        provenance = Provenance(
            model_id=SERVED_MODEL_ID,
            backend=self.backend_type,
            provider=self.provider,
            revision_or_api_model=self.revision_or_api_model,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            max_tokens=cfg.max_tokens,
            seed=cfg.seed,
            seed_honored="unknown",
            logprobs_available="logprobs" in choice and choice["logprobs"] is not None,
            quantization=self.quantization,
            generated_at=now_iso(),
            pool_id=self.pool_id,
        )
        return Sample(
            text=choice["message"]["content"],
            finish_reason=choice["finish_reason"],
            completion_tokens=usage.get("completion_tokens", 0),
            prompt_tokens=usage.get("prompt_tokens", 0),
            provenance=provenance,
            logprobs=choice.get("logprobs", {}).get("content") if choice.get("logprobs") else None,
        )


# Static-typing hint only (Protocol structural checks aren't enforced at
# runtime by this line) -- signals intent to type checkers, not a real
# runtime guarantee.
_: type[Backend] = HostedQwen35Backend
