"""
Day 12: real, live smoke test of bounded_beam_search on a handful of
real problems, before trusting it enough to run G5's real 20-problem
check. Checking exactly what was told to the user: real generated text
at each step, plausible PRM scores, sane budget numbers -- not just
"it didn't crash."
"""

import json
import sys
import urllib.request

sys.path.insert(0, "src")

from marginal_token.backends.base import DecodeConfig  # noqa: E402
from marginal_token.backends.hosted_endpoint import HostedQwen35Backend  # noqa: E402
from marginal_token.scoring.prm_client import HostedQwen25MathPRMClient  # noqa: E402
from marginal_token.search.beam import bounded_beam_search  # noqa: E402

PROBLEM_IDS = ["test/algebra/1031.json", "test/algebra/1072.json", "test/algebra/1184.json"]


def fetch(ids):
    wanted = set(ids)
    found = {}
    offset = 0
    while offset < 500 and len(found) < len(wanted):
        url = (f"https://datasets-server.huggingface.co/rows?dataset=HuggingFaceH4/MATH-500"
               f"&config=default&split=test&offset={offset}&length=100")
        with urllib.request.urlopen(url, timeout=30) as resp:
            d = json.load(resp)
        for r in d.get("rows", []):
            row = r["row"]
            if row["unique_id"] in wanted:
                found[row["unique_id"]] = (row["problem"], row["answer"])
        offset += 100
    return found


def main():
    problems = fetch(PROBLEM_IDS)
    backend = HostedQwen35Backend(pool_id="search_smoke_test")
    prm_client = HostedQwen25MathPRMClient()
    cfg = DecodeConfig(temperature=0.8, top_p=0.95, max_tokens=1024, thinking_mode=False)

    for pid in PROBLEM_IDS:
        problem_text, gold = problems[pid]
        print(f"\n{'='*70}\n{pid} (gold: {gold})\n{'='*70}")
        result = bounded_beam_search(
            problem_text, backend, prm_client, cfg,
            token_budget=1500, beam_width=2, max_steps=8, tokens_per_step_cap=256,
        )
        print(f"n_steps={result.n_steps}")
        print(f"charge: policy_tokens={result.charge.policy_tokens} "
              f"discarded_beam_tokens={result.charge.discarded_beam_tokens} "
              f"prm_forwards={result.charge.prm_forwards}")
        print(f"final text (last 500 chars):\n...{result.final_text[-500:]}")
        print(f"\nper-round mean_rewards seen:")
        for i, round_candidates in enumerate(result.beams_history):
            rewards = [round(b.mean_reward, 3) for b in round_candidates]
            print(f"  round {i}: {rewards}")


if __name__ == "__main__":
    main()
