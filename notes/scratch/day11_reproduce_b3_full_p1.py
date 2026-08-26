"""
Day 5's B3 comparison (plain majority vs PRM-weighted majority vs
PRM-argmax), rerun at the FULL 500-problem P1 scale -- previously only
checked on the 100-problem dev-100 subset (notes/2026-08-23.md). Pure
local analysis: P1 is fully generated (Day 6-9) and fully PRM-scored
(Day 10/11), so this needs zero new network calls, using the
productionized `PoolStore`/`PRMScoreStore`/`selectors` modules end to
end for the first time at this scale.
"""

import json
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, "src")

from marginal_token.answers.equivalence import check_equivalent  # noqa: E402
from marginal_token.answers.extraction import extract_answer  # noqa: E402
from marginal_token.answers.taxonomy import FailureStatus  # noqa: E402
from marginal_token.evaluation.stats import paired_bootstrap_bca  # noqa: E402
from marginal_token.pools.store import PoolStore  # noqa: E402
from marginal_token.scoring.pipeline import PRMScoreStore, compute_score_id  # noqa: E402
from marginal_token.scoring.prm_client import HostedQwen25MathPRMClient  # noqa: E402
from marginal_token.selectors.basic import VoteEntry, accuracy, plain_majority  # noqa: E402
from marginal_token.selectors.prm_based import WeightedVoteEntry, prm_argmax, prm_weighted_majority  # noqa: E402

import numpy as np

POOL_ROOT = "results/pools"
SCORE_ROOT = "results/scores"


def fetch_all_problem_data(unique_ids):
    wanted = set(unique_ids)
    found = {}
    offset = 0
    while offset < 500 and len(found) < len(wanted):
        url = (f"https://datasets-server.huggingface.co/rows?dataset=HuggingFaceH4/MATH-500"
               f"&config=default&split=test&offset={offset}&length=100")
        d = {}
        for _attempt in range(3):
            try:
                with urllib.request.urlopen(url, timeout=30) as resp:
                    d = json.load(resp)
                break
            except Exception:
                time.sleep(2)
        rows = d.get("rows", [])
        if not rows:
            break
        for r in rows:
            row = r["row"]
            if row["unique_id"] in wanted:
                found[row["unique_id"]] = row["answer"]
        offset += 100
    return found


def main():
    pool_meta = []
    for jsonl_path in sorted(Path(POOL_ROOT).glob("*/*.jsonl")):
        pool_id = jsonl_path.parent.name
        problem_id = jsonl_path.stem.replace("__", "/")
        if problem_id.isdigit():
            continue  # P2/OlympiadBench, shares this root -- not part of this analysis
        pool_meta.append((pool_id, problem_id))
    print(f"{len(pool_meta)} P1 (MATH-500) pools found")
    assert len(pool_meta) == 500, f"expected 500, got {len(pool_meta)}"

    gold = fetch_all_problem_data([pid for _, pid in pool_meta])
    print(f"fetched {len(gold)} gold answers")

    pool_store = PoolStore(POOL_ROOT)
    score_store = PRMScoreStore(SCORE_ROOT)
    prm_role = HostedQwen25MathPRMClient.role

    plain_results, weighted_results, argmax_results = [], [], []
    n_missing_scores = 0

    for pool_id, problem_id in pool_meta:
        pool = pool_store.load(pool_id, problem_id, "math500", "qwen3.5-4b", "policy_primary")
        assert len(pool) == 32

        score_id = compute_score_id(pool_id, prm_role, "double_newline")
        scores = {s.sample_idx: s for s in score_store.load(score_id, problem_id)}
        if len(scores) != 32:
            n_missing_scores += 1

        plain_votes, weighted_votes = [], []
        for sample in pool.samples:
            extraction = extract_answer(sample.text, finish_reason=sample.finish_reason)
            if extraction.status != FailureStatus.OK:
                plain_votes.append(VoteEntry(None, None))
                weighted_votes.append(WeightedVoteEntry(None, None, None))
                continue
            eq = check_equivalent(prediction=extraction.value, gold=gold[problem_id])
            key = str(extraction.value)
            is_correct = eq.equivalent
            plain_votes.append(VoteEntry(key, is_correct))
            score = scores.get(sample.sample_idx)
            weight = score.mean_reward if (score and score.status == FailureStatus.OK.value) else None
            weighted_votes.append(WeightedVoteEntry(key, is_correct, weight))

        plain_results.append(plain_majority(plain_votes))
        weighted_results.append(prm_weighted_majority(weighted_votes))
        argmax_results.append(prm_argmax(weighted_votes))

    if n_missing_scores:
        print(f"WARNING: {n_missing_scores} problems had incomplete PRM scores (expected 0 -- check before trusting)")

    plain_acc = accuracy(plain_results)
    weighted_acc = accuracy(weighted_results)
    argmax_acc = accuracy(argmax_results)

    print(f"\n{'selector':<26}{'n':>6}{'accuracy':>12}")
    print(f"{'plain_majority':<26}{500:>6}{plain_acc:>12.4f}")
    print(f"{'prm_weighted_majority':<26}{500:>6}{weighted_acc:>12.4f}")
    print(f"{'prm_argmax':<26}{500:>6}{argmax_acc:>12.4f}")

    diffs = np.array([
        (1 if w.is_correct else 0) - (1 if p.is_correct else 0)
        for p, w in zip(plain_results, weighted_results)
    ], dtype=float)
    boot = paired_bootstrap_bca(diffs, n_resamples=10_000, seed=42)
    print(f"\nPRM-weighted vs plain majority margin (full 500-problem P1): "
          f"{boot.point_estimate*100:+.2f}pp, 95% BCa CI [{boot.ci_lo*100:.2f}, {boot.ci_hi*100:.2f}]")

    n_flips = sum(1 for p, w in zip(plain_results, weighted_results) if p.is_correct != w.is_correct)
    print(f"individual-problem flips (plain vs weighted): {n_flips}/500")

    print("\n(Cross-check against the dev-100 subset finding, notes/2026-08-23.md: "
          "plain=0.730, weighted=0.730, 0 flips there. This is the FULL 500-problem P1 pool, "
          "a materially larger and non-identical sample -- not expected to match exactly, "
          "but should be directionally consistent if the underlying finding is robust.)")


if __name__ == "__main__":
    main()
