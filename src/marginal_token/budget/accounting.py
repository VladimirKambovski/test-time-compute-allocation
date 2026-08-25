"""Exact token/latency accounting. Charges policy tokens generated, PRM
forward passes, AND discarded beam-search branches -- undercounting any
of these invalidates every matched-token comparison in the project
(CLAUDE.md invariant #4).

SAMPLE and SELECT receive the SAME budget B (docs/brief.md §4). Because
PRM forwards are not free, SELECT buys FEWER raw samples than SAMPLE at
equal B -- this is load-bearing, not incidental: it's what makes
"sample more" vs. "select better" a genuine matched-token tradeoff
rather than SELECT getting a free bonus. If SELECT ever buys the same or
more samples than SAMPLE at equal B, the budget split is implemented
wrong.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Charge:
    """Every token/forward actually spent, regardless of whether it
    contributed to the final answer -- a discarded beam branch still
    cost real compute and must be charged (invariant #4).
    """

    policy_tokens: int = 0
    prm_forwards: int = 0
    discarded_beam_tokens: int = 0

    def total_token_equivalent(self, prm_forward_cost_tokens: float) -> float:
        """PRM forwards don't consume "policy tokens" directly, but they
        cost real compute -- converted to a token-equivalent so SAMPLE
        and SELECT can be compared on one matched-token axis.
        """
        return self.policy_tokens + self.discarded_beam_tokens + self.prm_forwards * prm_forward_cost_tokens


def charge_sample_action(n_samples: int, tokens_per_sample: int) -> Charge:
    """A1 SAMPLE: spend the entire budget on more policy samples, plain
    majority. No PRM involvement at all -- prm_forwards must be exactly 0.
    """
    return Charge(policy_tokens=n_samples * tokens_per_sample, prm_forwards=0, discarded_beam_tokens=0)


def budget_split_for_select(
    budget_b: int, tokens_per_sample: int, prm_forward_cost_tokens: float, prm_forwards_per_sample: int = 1,
) -> int:
    """How many samples can SELECT afford under the SAME budget B that
    SAMPLE gets, once PRM scoring is paid for out of that same budget?
    Always <= what SAMPLE could buy with the identical B, strictly less
    whenever PRM scoring costs anything -- this is the core tradeoff
    invariant #4 exists to make possible to test.
    """
    cost_per_sample = tokens_per_sample + prm_forwards_per_sample * prm_forward_cost_tokens
    if cost_per_sample <= 0:
        raise ValueError("cost_per_sample must be positive")
    return int(budget_b // cost_per_sample)


def charge_select_action(
    budget_b: int, tokens_per_sample: int, prm_forward_cost_tokens: float, prm_forwards_per_sample: int = 1,
) -> Charge:
    """A2 SELECT: budget B split between more samples and PRM scoring,
    then PRM-weighted selection. Charged tokens = policy tokens for the
    (fewer) samples bought + the PRM forwards spent scoring them.
    """
    n_samples = budget_split_for_select(budget_b, tokens_per_sample, prm_forward_cost_tokens, prm_forwards_per_sample)
    return Charge(
        policy_tokens=n_samples * tokens_per_sample,
        prm_forwards=n_samples * prm_forwards_per_sample,
        discarded_beam_tokens=0,
    )


def charge_search_action(
    kept_tokens: int, discarded_branch_token_counts: list[int], prm_forwards: int,
) -> Charge:
    """A3 SEARCH: policy tokens INCLUDING discarded beam branches, plus
    PRM forwards for guiding the search. Every discarded branch's tokens
    must be counted -- a beam that was generated and then pruned still
    cost real compute; dropping it from the charge would make SEARCH
    look artificially cheap relative to SAMPLE/SELECT.
    """
    return Charge(
        policy_tokens=kept_tokens,
        prm_forwards=prm_forwards,
        discarded_beam_tokens=sum(discarded_branch_token_counts),
    )
