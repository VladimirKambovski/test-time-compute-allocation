"""
Gate G4 (Day 9). On the local-weights backend, regenerating a small pool
with the identical config must produce byte-identical output, verified
by content hash.

On an API backend, seeds/determinism are often not honoured — that's
expected and OK. In that case this test should record
`determinism: not guaranteed by backend` rather than fail the build.
The frozen pool artifact + its hash is the reproducibility unit in that
case (see docs/brief.md section 27.2), NOT byte-level regeneration.

**G4 verdict, decided here (2026-08-23):** local-backend byte-identity is
not exercised, honestly, not faked. `configs/policies/qwen3.5-4b.yaml`
lists `local_vllm` as `confirmed_available` (Day 1's G0 documentation
check), but zero real generation has ever actually run against it --
the raw GPU servers exist but SSH access was never resolved
(`notes/2026-08-22.md`, `configs/backends/hosted-endpoints.yaml`'s
`raw_gpu_servers` entry). Testing "does local vLLM regenerate
byte-identically" against a backend that has never generated a single
real sample would be either a vacuous pass or an outright fabrication --
neither is honest. Skipped with a dated, specific reason instead of left
as a permanent `NotImplementedError`, per the same discipline Day 3 used
for a real blocker (`pytest.mark.skipif`, not a silent stub). Revisit
only if/when local-weights generation is actually exercised (the SSH
question, or a future rented GPU).

The API-backend half is real, checkable work now: every real sample
P1 has generated so far (`results/pools/`, live as of Day 6) carries a
genuine `Provenance` block with `seed_honored="unknown"` (per
`HostedQwen35Backend.capabilities()`) -- this test asserts that's
actually true of real, on-disk samples, not a synthetic stand-in.
"""

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
POOL_ROOT = REPO_ROOT / "results/pools"


def _any_real_provenance_dicts(limit: int = 50) -> list[dict]:
    """Read `provenance` blocks straight off real, on-disk pool files --
    no need to reconstruct full `Sample` objects for a check that only
    looks at two provenance fields.
    """
    provenances: list[dict] = []
    if not POOL_ROOT.exists():
        return provenances
    for jsonl_path in POOL_ROOT.glob("*/*.jsonl"):
        with open(jsonl_path) as f:
            for line in f:
                if line.strip():
                    provenances.append(json.loads(line)["provenance"])
                    if len(provenances) >= limit:
                        return provenances
    return provenances


@pytest.mark.skip(
    reason="Local-weights backend (local_vllm) has never actually generated a real sample -- "
           "raw GPU SSH access unresolved as of 2026-08-22 (notes/2026-08-22.md). Testing "
           "byte-identity against a backend that has never run would be a vacuous pass, not a "
           "real determinism check. Un-skip only once local generation is genuinely exercised."
)
def test_local_backend_determinism():
    raise NotImplementedError


def test_api_backend_records_nondeterminism_honestly():
    provenances = _any_real_provenance_dicts()
    if not provenances:
        pytest.skip("No real samples on disk yet -- P1 generation (Day 6) may not have produced "
                    "any complete files at the moment this test ran. Not a backend-honesty failure.")
    api_provenances = [p for p in provenances if p["backend"] == "api_host"]
    assert api_provenances, "expected at least one real api_host sample among the on-disk pools"
    for prov in api_provenances:
        # The honesty check itself: an API backend must never silently
        # claim determinism it can't back up. `seed_honored` must be
        # exactly "unknown" or False -- never True, since bit-reproducibility
        # across repeated calls to this hosted endpoint has never been
        # confirmed (configs/backends/hosted-endpoints.yaml's own
        # verification log says so explicitly).
        assert prov["seed_honored"] in ("unknown", False), (
            f"api_host sample claims seed_honored={prov['seed_honored']!r} -- "
            f"never verified, must not be silently reported as True"
        )
