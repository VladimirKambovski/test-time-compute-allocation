"""
Every generated sample must carry full provenance, and pools must never
silently mix backends, providers, model revisions, or decode configs.
See CLAUDE.md invariant #3 and docs/brief.md section 27.6.

Uses only synthetic Sample/Provenance objects and the existing Day 4
pool data already on disk -- no new generation.
"""

import pytest

from marginal_token.backends.base import Provenance, Sample, validate_provenance
from marginal_token.pools.store import Pool, PoolContractError, PoolManifest


def _provenance(**overrides):
    defaults = dict(
        model_id="unsloth/Qwen3.5-4B-GGUF:BF16",
        backend="api_host",
        provider="mentor_hosted_llamacpp",
        revision_or_api_model="unsloth/Qwen3.5-4B-GGUF:BF16",
        temperature=0.8,
        top_p=0.95,
        max_tokens=1024,
        seed=None,
        seed_honored="unknown",
        logprobs_available=True,
        quantization="bf16_gguf_third_party_conversion",
        generated_at="2026-08-21T00:00:00Z",
        pool_id="deadbeef",
    )
    defaults.update(overrides)
    return Provenance(**defaults)


def _sample(**prov_overrides):
    return Sample(
        text="\\boxed{42}",
        finish_reason="stop",
        completion_tokens=10,
        prompt_tokens=20,
        provenance=_provenance(**prov_overrides),
    )


def test_every_sample_has_full_provenance():
    # A complete provenance dict validates without raising.
    validate_provenance(_provenance().as_dict())

    # Missing any single required field must fail loudly, not silently pass.
    complete = _provenance().as_dict()
    for field in complete:
        incomplete = {k: v for k, v in complete.items() if k != field}
        with pytest.raises(ValueError):
            validate_provenance(incomplete)


def test_pool_validator_rejects_contract_mismatch():
    manifest = PoolManifest(policy_ref="qwen3.5-4b", backend_ref="hosted_endpoint",
                             benchmark_id="math500", contract_key=None)
    pool = Pool(problem_id="test/algebra/1.json", pool_id="deadbeef", manifest=manifest)

    first = _sample()
    first.sample_idx = 0
    pool.add(first)  # establishes the manifest's contract_key

    # Same contract fields, different sample_idx -- must be accepted.
    same_contract = _sample()
    same_contract.sample_idx = 1
    pool.add(same_contract)
    assert len(pool) == 2

    # Different max_tokens (a contract field) -- must be REJECTED (raised),
    # never silently dropped or silently merged into the pool.
    mismatched = _sample(max_tokens=2048)
    mismatched.sample_idx = 2
    with pytest.raises(PoolContractError):
        pool.add(mismatched)
    assert len(pool) == 2  # rejection must not have silently appended it

    # Different backend/provider -- also a contract field, also rejected.
    different_backend = _sample(backend="local_vllm", provider="local")
    with pytest.raises(PoolContractError):
        pool.add(different_backend)

    # Different quantization -- also rejected per §27.6 ("quantization /
    # numeric precision" is explicitly contract-relevant).
    different_quant = _sample(quantization="awq4")
    with pytest.raises(PoolContractError):
        pool.add(different_quant)

    # But seed/seed_honored/generated_at/logprobs_available are explicitly
    # "permitted to differ within one pool" per §27.6 -- must NOT raise.
    different_seed = _sample(seed=999, seed_honored=True, generated_at="2026-08-21T01:00:00Z")
    different_seed.sample_idx = 3
    pool.add(different_seed)
    assert len(pool) == 3
