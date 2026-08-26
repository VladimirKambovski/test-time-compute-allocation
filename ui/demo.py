"""
Day 17: the demo, benchmark mode. "Someone can pick a problem and watch
the full decision path to an outcome" (roadmap's exact done-when bar),
reading ONLY cached artifacts (pools, PRM scores, the frozen Detective
model) -- zero GPU calls, zero policy/PRM API calls. The one network
call this makes (fetching a problem's gold answer for display) is a
dataset metadata lookup, not inference, same as every analysis script
in notes/scratch/ tonight.

Deliberately minimal, not the full ui/ panel set (probe/evidence/
controller/spend/outcome panels, comparison mode) docs/brief.md
describes as SHOULD-tier -- scoped down under the 3-4-day deadline per
explicit instruction to protect reproducibility/held-out/report time
over demo polish. This meets the roadmap's actual MUST bar for Day 17;
it is not the full brief-described UI.

Same discipline as gateway/app.py and replay/engine.py: this script
contains NO allocation logic of its own (invariant #5) -- it calls
controller.decide() exactly once per problem, via replay.replay_one(),
same as the live gateway would. Everything after that is OUTCOME
SIMULATION from cached data (what would have happened, using the real
cached samples/scores), not a second decision.

Usage:
    python ui/demo.py test/algebra/2176.json
    python ui/demo.py 1606 --benchmark olympiad-a
    python ui/demo.py --random
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import joblib  # noqa: E402

from marginal_token.answers.equivalence import check_equivalent  # noqa: E402
from marginal_token.answers.extraction import extract_answer  # noqa: E402
from marginal_token.answers.taxonomy import FailureStatus  # noqa: E402
from marginal_token.budget.accounting import budget_split_for_select, charge_sample_action, charge_select_action  # noqa: E402
from marginal_token.controller.base import Budget, Probe  # noqa: E402
from marginal_token.controller.features import featurize  # noqa: E402
from marginal_token.pools.store import PoolStore, compute_pool_id  # noqa: E402
from marginal_token.replay.engine import replay_one  # noqa: E402
from marginal_token.scoring.pipeline import PRMScoreStore, compute_score_id  # noqa: E402
from marginal_token.selectors.prm_based import WeightedVoteEntry, prm_weighted_majority  # noqa: E402

FROZEN_MODEL_PATH = "results/models/detective_frozen.joblib"
POOL_ROOT = "results/pools"
SCORE_ROOT = "results/scores"


def _fetch_gold(pid: str, benchmark: str) -> tuple[str, str]:
    """Metadata lookup only (no inference) -- returns (problem_text, gold)."""
    if benchmark == "math500":
        offset = 0
        while offset < 500:
            url = (f"https://datasets-server.huggingface.co/rows?dataset=HuggingFaceH4/MATH-500"
                   f"&config=default&split=test&offset={offset}&length=100")
            d = json.load(urllib.request.urlopen(url, timeout=30))
            for r in d["rows"]:
                row = r["row"]
                if row["unique_id"] == pid:
                    return row["problem"], row["answer"]
            offset += 100
        raise ValueError(f"{pid} not found in MATH-500")
    else:
        offset = 0
        while offset < 674:
            url = (f"https://datasets-server.huggingface.co/rows?dataset=Hothan/OlympiadBench"
                   f"&config=OE_TO_maths_en_COMP&split=train&offset={offset}&length=100")
            d = json.load(urllib.request.urlopen(url, timeout=30))
            for r in d["rows"]:
                row = r["row"]
                if str(row["id"]) == pid:
                    return row["question"], row["final_answer"][0]
            offset += 100
        raise ValueError(f"{pid} not found in OlympiadBench-A")


def _pick_random_problem() -> tuple[str, str]:
    ids = [str(x) for x in json.load(open("configs/benchmarks/data/olympiad-a-ids.json"))]
    if random.random() < 0.5:
        math_ids = json.load(open("configs/benchmarks/data/math500-dev100-ids.json"))
        return random.choice(math_ids), "math500"
    return random.choice(ids), "olympiad-a"


def _outcome_for_action(action: str) -> str:
    if action == "abstain":
        return "declined"
    if action in ("sample", "select"):
        return "escalated"
    return "answered"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("problem_id", nargs="?", help="e.g. test/algebra/2176.json or 1606")
    parser.add_argument("--benchmark", choices=["math500", "olympiad-a"], default="math500")
    parser.add_argument("--budget", type=int, default=32, help="budget level (samples), one of {2,4,8,16,32}")
    parser.add_argument("--random", action="store_true", help="pick a random dev problem")
    args = parser.parse_args()

    if args.random or not args.problem_id:
        pid, benchmark = _pick_random_problem()
    else:
        pid, benchmark = args.problem_id, args.benchmark

    print(f"{'='*70}\nCOMPUTE-AWARE REASONING GATEWAY -- benchmark mode (cached artifacts only)\n{'='*70}")
    print(f"\nProblem: {pid}  ({benchmark})")

    problem_text, gold = _fetch_gold(pid, benchmark)
    print(f"Q: {problem_text[:300]}{'...' if len(problem_text) > 300 else ''}")
    print(f"(gold answer, shown here for benchmark mode only -- a live deployment would not have this): {gold}")

    pool_store = PoolStore(POOL_ROOT)
    pool_id = compute_pool_id(policy_ref="qwen3.5-4b", backend_ref="policy_primary", benchmark_id=benchmark,
                                problem_id=pid, temperature=0.8, top_p=0.95, max_tokens=1024, seed=None, n=32)
    pool = pool_store.load(pool_id, pid, benchmark, "qwen3.5-4b", "policy_primary")
    ordered = sorted(pool.samples, key=lambda s: s.sample_idx)

    score_store = PRMScoreStore(SCORE_ROOT)
    score_id = compute_score_id(pool_id, "primary_prm", "double_newline")
    try:
        scores = {sc.sample_idx: sc for sc in score_store.load(score_id, pid)}
    except Exception:
        scores = {}

    # --- 1. THE FREE PROBE (k=4) ---
    probe = Probe(samples=ordered[:4])
    print(f"\n--- Free probe (k=4 samples, already paid for) ---")
    for s in probe.samples:
        ext = extract_answer(s.text, finish_reason=s.finish_reason)
        shown = ext.value if ext.status == FailureStatus.OK else f"[{ext.status}]"
        print(f"  sample {s.sample_idx}: answer={shown}")

    # --- 2. THE DECISION (the ONLY allocation-logic call) ---
    controller = joblib.load(FROZEN_MODEL_PATH)
    budget = Budget(max_tokens=args.budget, max_latency_ms=None)
    result = replay_one(controller, problem_id=pid, probe=probe, budget=budget)
    decision = result.decision
    outcome = _outcome_for_action(decision.action)

    print(f"\n--- Controller decision ---")
    print(f"  action: {decision.action}")
    print(f"  confidence (class probabilities): " +
          ", ".join(f"{k}={v:.3f}" for k, v in sorted(decision.class_probs.items(), key=lambda kv: -kv[1]) if v > 0.001))
    print(f"  budget granted: {decision.budget_grant}")

    # --- 3. OUTCOME SIMULATION (from cached data, zero new inference) ---
    print(f"\n--- Outcome ({outcome}) ---")
    # NOTE: vote-counting groups by str(ext.value) (a fine, common way to
    # group identical answers), but check_equivalent needs the ORIGINAL,
    # non-stringified parsed value -- math_verify does type-sensitive
    # symbolic comparison, and stringifying loses that. Every branch
    # below keeps a str-key -> original-value map for exactly this reason
    # (caught live: an early version stringified before the equivalence
    # check and silently reported "sqrt(3) != $\sqrt{3}$" as incorrect
    # when it's actually a real match -- verified against
    # oracle_labels._majority_correct, which does this correctly, before
    # trusting the fix).

    if decision.action == "stop":
        counts, by_key = {}, {}
        for s in probe.samples:
            ext = extract_answer(s.text, finish_reason=s.finish_reason)
            if ext.status == FailureStatus.OK:
                key = str(ext.value)
                counts[key] = counts.get(key, 0) + 1
                by_key[key] = ext.value
        top_key = max(counts, key=counts.get) if counts else None
        answer, answer_value = top_key, (by_key.get(top_key) if top_key else None)
        print(f"  answer (majority of the free probe): {answer}")
        print(f"  cost: 0 additional tokens (already paid for by the probe)")

    elif decision.action == "sample":
        b = args.budget
        subset = ordered[:b]
        counts, by_key = {}, {}
        for s in subset:
            ext = extract_answer(s.text, finish_reason=s.finish_reason)
            if ext.status == FailureStatus.OK:
                key = str(ext.value)
                counts[key] = counts.get(key, 0) + 1
                by_key[key] = ext.value
        top_key = max(counts, key=counts.get) if counts else None
        answer, answer_value = top_key, (by_key.get(top_key) if top_key else None)
        charge = charge_sample_action(n_samples=b, tokens_per_sample=int(sum(s.completion_tokens for s in subset) / b))
        print(f"  answer (majority of {b} cached samples): {answer}")
        print(f"  cost: {charge.policy_tokens} tokens ({b} samples replayed from the cached pool)")

    elif decision.action == "select":
        b = args.budget
        mean_len = sum(s.completion_tokens for s in ordered) / len(ordered)
        n_select = budget_split_for_select(b * mean_len, mean_len, mean_len)  # disclosed assumption, see notes/2026-08-26.md
        votes, by_key = [], {}
        for s in ordered[:n_select]:
            ext = extract_answer(s.text, finish_reason=s.finish_reason)
            sc = scores.get(s.sample_idx)
            weight = sc.mean_reward if (sc is not None and sc.status == "ok") else None
            key = str(ext.value) if ext.status == FailureStatus.OK else None
            if key is not None:
                by_key[key] = ext.value
            votes.append(WeightedVoteEntry(answer_key=key, is_correct=None, weight=weight))
        result_v = prm_weighted_majority(votes)
        answer = result_v.winning_key
        answer_value = by_key.get(answer) if answer else None
        charge = charge_select_action(budget_b=int(b * mean_len), tokens_per_sample=int(mean_len), prm_forward_cost_tokens=mean_len)
        print(f"  answer (PRM-weighted majority of {n_select} cached samples, PRM cost eats into the same budget): {answer}")
        print(f"  cost: {charge.policy_tokens} policy tokens + {charge.prm_forwards} PRM forwards")

    else:  # abstain
        answer, answer_value = None, None
        print(f"  declined -- reason: controller_predicted_abstain")
        print(f"  cost: 0 tokens (no budget spent past the free probe)")

    if answer_value is not None:
        eq = check_equivalent(prediction=answer_value, gold=gold)
        print(f"\n  correct (benchmark mode only, checked against gold): {eq.equivalent}")
    else:
        print(f"\n  no answer returned to check.")

    print(f"\n{'='*70}\nzero GPU calls, zero policy/PRM API calls -- everything above replayed from cached artifacts.\n{'='*70}")


if __name__ == "__main__":
    main()
