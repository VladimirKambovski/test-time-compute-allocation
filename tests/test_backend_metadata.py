"""
Every generated sample must carry full provenance, and pools must never
silently mix backends, providers, model revisions, or decode configs.
See CLAUDE.md invariant #3 and docs/brief.md section 27.6.

TODO:
- assert every sample in a pool has: model_id, backend, provider,
  revision_or_api_model, temperature, top_p, max_tokens, seed,
  seed_honored, logprobs_available, generated_at, pool_id
- assert a pool validator rejects (raises, does not silently drop) a
  sample whose contract fields (model revision, quantization, temp,
  top_p, top_k, max_tokens, prompt template, stop sequences, backend/
  provider deployment) disagree with the pool's manifest
"""


def test_every_sample_has_full_provenance():
    raise NotImplementedError


def test_pool_validator_rejects_contract_mismatch():
    raise NotImplementedError
