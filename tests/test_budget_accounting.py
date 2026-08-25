"""
Exact token accounting is load-bearing for every matched-token claim
in the project (H1, H4). Must charge: policy tokens generated, PRM
forward passes, AND discarded beam-search branches.
"""

from marginal_token.budget.accounting import (
    charge_sample_action,
    charge_search_action,
    charge_select_action,
)


def test_sample_action_charges_only_policy_tokens():
    charge = charge_sample_action(n_samples=32, tokens_per_sample=800)
    assert charge.policy_tokens == 32 * 800
    assert charge.prm_forwards == 0
    assert charge.discarded_beam_tokens == 0


def test_select_action_buys_fewer_samples_than_sample_at_equal_budget():
    budget_b = 32 * 800  # the exact budget SAMPLE spends on 32 samples above
    tokens_per_sample = 800
    prm_forward_cost_tokens = 200  # a PRM forward pass costs the equivalent of 200 policy tokens

    sample_charge = charge_sample_action(
        n_samples=budget_b // tokens_per_sample, tokens_per_sample=tokens_per_sample
    )
    select_charge = charge_select_action(
        budget_b=budget_b, tokens_per_sample=tokens_per_sample, prm_forward_cost_tokens=prm_forward_cost_tokens
    )

    n_sample_equivalent = sample_charge.policy_tokens // tokens_per_sample
    n_select_equivalent = select_charge.policy_tokens // tokens_per_sample

    # The core SAMPLE-vs-SELECT tradeoff: SELECT strictly buys fewer raw
    # samples than SAMPLE at the identical budget, because PRM forwards
    # aren't free. If these were equal, the budget split would be wrong.
    assert n_select_equivalent < n_sample_equivalent
    assert select_charge.prm_forwards == n_select_equivalent  # one PRM forward per sample bought

    # Both actions must stay within the same total budget B.
    assert sample_charge.total_token_equivalent(prm_forward_cost_tokens) <= budget_b
    assert select_charge.total_token_equivalent(prm_forward_cost_tokens) <= budget_b


def test_select_action_prm_forwards_never_zero_when_samples_bought():
    # A degenerate implementation could accidentally buy samples for
    # SELECT without ever charging the PRM forwards that selecting is
    # supposed to cost -- that would silently make SELECT look as cheap
    # as SAMPLE and invalidate the whole tradeoff.
    charge = charge_select_action(budget_b=10_000, tokens_per_sample=800, prm_forward_cost_tokens=200)
    assert charge.policy_tokens > 0
    assert charge.prm_forwards > 0


def test_search_action_charges_discarded_beam_tokens():
    kept_tokens = 1200
    discarded = [300, 450, 150]  # three pruned beam branches
    prm_forwards = 8

    charge = charge_search_action(
        kept_tokens=kept_tokens, discarded_branch_token_counts=discarded, prm_forwards=prm_forwards
    )

    assert charge.policy_tokens == kept_tokens
    assert charge.discarded_beam_tokens == sum(discarded)
    assert charge.prm_forwards == prm_forwards

    # The whole point: total charge must include the discarded tokens,
    # not just what ended up in the final kept path. Silently omitting
    # them would make SEARCH look cheaper than it actually was.
    total = charge.total_token_equivalent(prm_forward_cost_tokens=200)
    assert total == kept_tokens + sum(discarded) + prm_forwards * 200
    assert total > kept_tokens + prm_forwards * 200  # i.e. discarded tokens actually counted
