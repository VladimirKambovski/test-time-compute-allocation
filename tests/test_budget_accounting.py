"""
Exact token accounting is load-bearing for every matched-token claim
in the project (H1, H4). Must charge: policy tokens generated, PRM
forward passes, AND discarded beam-search branches.

TODO:
- construct a synthetic SAMPLE action: assert charged tokens == sum of
  generated policy tokens, PRM forwards == 0
- construct a synthetic SELECT action: assert charged tokens == policy
  tokens + PRM forward count, and that SELECT's sample count is lower
  than SAMPLE's at the same budget B (this is the core SAMPLE-vs-SELECT
  tradeoff — if they're equal, the budget split is implemented wrong)
- construct a synthetic SEARCH action with at least one discarded beam
  branch: assert the discarded branch's tokens are included in the charge
"""


def test_sample_action_charges_only_policy_tokens():
    raise NotImplementedError


def test_select_action_buys_fewer_samples_than_sample_at_equal_budget():
    raise NotImplementedError


def test_search_action_charges_discarded_beam_tokens():
    raise NotImplementedError
