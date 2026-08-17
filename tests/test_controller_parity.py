"""
THE most important test in the repo. Asserts that the offline replay
engine and the live gateway produce the SAME decision for the SAME
probe evidence, because they must call the identical Controller object.

This is what makes the demo an operationalization of the research
result rather than a dashboard wrapped around it. Do not weaken or
skip this test to make integration easier — if it's hard to pass,
that means gateway and replay have diverged, which is the actual bug.

TODO:
- construct one fixed Probe (4 samples + features) as a fixture
- call replay_engine's controller.decide(probe, budget) directly
- call gateway's /solve endpoint (or its internal controller call path)
  with equivalent inputs
- assert the returned Decision (action, budget_grant, class_probs) is
  identical between the two paths
- run this for at least one case per action (STOP, SAMPLE, SELECT,
  SEARCH, ABSTAIN)
"""


def test_replay_and_gateway_agree_on_probe_decision():
    raise NotImplementedError
