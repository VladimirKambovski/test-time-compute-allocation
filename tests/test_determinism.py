"""
Gate G4 (Day 9). On the local-weights backend, regenerating a small pool
with the identical config must produce byte-identical output, verified
by content hash.

On an API backend, seeds/determinism are often not honoured — that's
expected and OK. In that case this test should record
`determinism: not guaranteed by backend` rather than fail the build.
The frozen pool artifact + its hash is the reproducibility unit in that
case (see docs/brief.md section 27.2), NOT byte-level regeneration.

TODO:
- generate a small pool (e.g. 5 problems, N=4) twice with identical config
- if backend is local_vllm: assert hash(run_1) == hash(run_2)
- if backend is api_host: assert the metadata block records
  seed_honored=false or "unknown", and skip the byte-identity assertion
  without failing the test
"""


def test_local_backend_determinism():
    raise NotImplementedError


def test_api_backend_records_nondeterminism_honestly():
    raise NotImplementedError
