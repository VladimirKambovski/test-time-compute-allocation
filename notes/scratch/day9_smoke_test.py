"""
Day 9: full smoke test on 50 real problems -- generate (already done by
P1, running since Day 6) -> score (Day 7's scoring/) -> select (Day 8's
selectors/) -> metrics (evaluation/stats.py) -> one figure, entirely
through the productionized modules, on genuinely real data.

Not the real infra's job to orchestrate this end-to-end yet (that's
Day 10's controller/replay wiring) -- this is a one-off smoke script in
the same spirit as Day 4/5's scratch scripts, proving the pieces already
built actually compose correctly over real data before anything gets
wired together permanently.
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
from marginal_token.scoring.pipeline import PRMScoreStore, score_pool  # noqa: E402
from marginal_token.scoring.prm_client import HostedQwen25MathPRMClient  # noqa: E402
from marginal_token.selectors.basic import VoteEntry, accuracy, oracle_pass_at_k, plain_majority  # noqa: E402
from marginal_token.selectors.prm_based import WeightedVoteEntry, prm_argmax, prm_weighted_majority  # noqa: E402

import numpy as np

POOL_ROOT = Path("results/pools")
N_SMOKE = 50
BENCHMARK_ID = "math500"
POLICY_REF = "qwen3.5-4b"
BACKEND_REF = "policy_primary"


def find_complete_pools(n: int) -> list[tuple[str, str]]:
    """Scan results/pools/ for (pool_id, problem_id) pairs with exactly
    32 real samples -- P1's real, currently-still-running output, not
    synthetic. Deterministic: sorted by problem_id, first `n`.
    """
    found = []
    for jsonl_path in sorted(POOL_ROOT.glob("*/*.jsonl")):
        with open(jsonl_path) as f:
            n_lines = sum(1 for line in f if line.strip())
        if n_lines == 32:
            pool_id = jsonl_path.parent.name
            problem_id = jsonl_path.stem.replace("__", "/")
            found.append((pool_id, problem_id))
    found.sort(key=lambda t: t[1])
    return found[:n]


def fetch_problem_text_and_gold(unique_ids: list[str]) -> dict[str, tuple[str, str]]:
    """{unique_id: (problem_text, gold_answer)} via the public HF
    datasets-server REST API -- same pattern as day4/5/run_sweeps.py.
    """
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
                found[row["unique_id"]] = (row["problem"], row["answer"])
        offset += 100
    return found


def main():
    pools_meta = find_complete_pools(N_SMOKE)
    assert len(pools_meta) == N_SMOKE, f"expected {N_SMOKE} complete pools, found {len(pools_meta)}"
    print(f"smoke test: {N_SMOKE} real, complete P1 pools selected (deterministic, sorted by problem_id)")

    problem_ids = [pid for _, pid in pools_meta]
    problem_data = fetch_problem_text_and_gold(problem_ids)
    missing = set(problem_ids) - set(problem_data)
    assert not missing, f"could not fetch problem text/gold for: {missing}"

    pool_store = PoolStore(str(POOL_ROOT))
    score_store = PRMScoreStore("results/scores")
    prm_client = HostedQwen25MathPRMClient()

    plain_results = []
    weighted_results = []
    argmax_results = []
    oracle_pass32 = []
    n_new_scores = 0

    t0 = time.time()
    for pool_id, problem_id in pools_meta:
        pool = pool_store.load(pool_id, problem_id, BENCHMARK_ID, POLICY_REF, BACKEND_REF)
        assert len(pool) == 32, f"{problem_id}: expected 32 samples, got {len(pool)}"
        problem_text, gold = problem_data[problem_id]

        # SCORE (resumable -- already-scored samples from Day 7's smoke
        # test are skipped automatically, only new ones cost a real call).
        new = score_pool(pool, query=problem_text, client=prm_client, store=score_store)
        n_new_scores += len(new)
        score_id_scores = {s.sample_idx: s for s in score_store.load(
            __import__("marginal_token.scoring.pipeline", fromlist=["compute_score_id"]).compute_score_id(
                pool_id, prm_client.role, "double_newline"), problem_id)}

        # SELECT: build vote entries from real extraction + equivalence
        # checks against the real gold answer, and real PRM weights.
        plain_votes, weighted_votes = [], []
        for sample in pool.samples:
            extraction = extract_answer(sample.text, finish_reason=sample.finish_reason)
            if extraction.status != FailureStatus.OK:
                plain_votes.append(VoteEntry(None, None))
                weighted_votes.append(WeightedVoteEntry(None, None, None))
                continue
            eq = check_equivalent(prediction=extraction.value, gold=gold)
            key = str(extraction.value)
            is_correct = eq.equivalent
            plain_votes.append(VoteEntry(key, is_correct))
            score = score_id_scores.get(sample.sample_idx)
            weight = score.mean_reward if (score and score.status == FailureStatus.OK.value) else None
            weighted_votes.append(WeightedVoteEntry(key, is_correct, weight))

        plain_results.append(plain_majority(plain_votes))
        weighted_results.append(prm_weighted_majority(weighted_votes))
        argmax_results.append(prm_argmax(weighted_votes))

        n_correct = sum(1 for v in plain_votes if v.is_correct)
        oracle_pass32.append(oracle_pass_at_k(plain_votes, 32))

    elapsed = time.time() - t0
    print(f"scored {n_new_scores} new samples this run, {elapsed:.1f}s")

    # METRICS
    plain_acc = accuracy(plain_results)
    weighted_acc = accuracy(weighted_results)
    argmax_acc = accuracy(argmax_results)
    oracle_ceiling = float(np.mean(oracle_pass32))

    print(f"\n{'selector':<30}{'accuracy':>12}")
    print(f"{'plain_majority':<30}{plain_acc:>12.4f}")
    print(f"{'prm_weighted_majority':<30}{weighted_acc:>12.4f}")
    print(f"{'prm_argmax':<30}{argmax_acc:>12.4f}")
    print(f"{'oracle_pass@32 (ceiling)':<30}{oracle_ceiling:>12.4f}")

    diffs = np.array([
        (1 if w.is_correct else 0) - (1 if p.is_correct else 0)
        for p, w in zip(plain_results, weighted_results)
    ], dtype=float)
    boot = paired_bootstrap_bca(diffs, n_resamples=10_000, seed=42)
    print(f"\nPRM-weighted vs plain majority margin (50-problem smoke slice): "
          f"{boot.point_estimate*100:+.2f}pp, 95% BCa CI [{boot.ci_lo*100:.2f}, {boot.ci_hi*100:.2f}]")
    print("(A 50-problem slice, not the full 500-problem P1 -- this is a smoke test of the "
          "pipeline, not a new headline number. Full-scale numbers wait for P1 to complete.)")

    # ONE FIGURE
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = ["plain\nmajority", "PRM-weighted\nmajority", "PRM\nargmax", "oracle\npass@32"]
    values = [plain_acc, weighted_acc, argmax_acc, oracle_ceiling]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(labels, values, color=["#4C72B0", "#55A868", "#C44E52", "#8172B2"])
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)
    ax.set_ylabel("accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Day 9 smoke test: selector accuracy, N=32, {N_SMOKE} real MATH-500 problems\n"
                  f"(qwen3.5-4b via hosted endpoint, primary PRM rung 1)", fontsize=10)
    fig.tight_layout()

    out_path = Path("results/figures")
    out_path.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path / "day9_smoke_test_selector_accuracy.png", dpi=150)
    print(f"\nfigure saved: {out_path / 'day9_smoke_test_selector_accuracy.png'}")


if __name__ == "__main__":
    main()
