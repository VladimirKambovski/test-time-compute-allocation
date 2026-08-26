"""
Day 15: E7 (7 policies x 5 matched-token budgets over cached pools,
evaluates H4) and E8 (cost-per-correct-answer). Real, out-of-fold
Detective/Fortune Teller predictions (same grouped-by-problem,
stratified-by-benchmark 5-fold CV, same random_state as
day13_e5_predictor_cv.py / day14_e5_full_comparators.py -- refit here
rather than reloaded, since earlier scripts didn't persist oof
predictions to disk; identical methodology, not a new/different fit).

**Disclosed modeling assumption, no documented value exists anywhere in
docs/brief.md or configs/prms/ for this:** a PRM forward pass is
assumed to cost the SAME token-equivalent as one policy sample
generation (prm_forward_cost_tokens = that problem's own mean
completion length). This is a deliberately conservative, easy-to-state
convention (not a measured number) -- under it, SELECT affords roughly
half as many raw samples as SAMPLE at equal budget B (one "slot" buys
1 sample + 1 score at ~2x a sample's cost each). Flagged for the
mentor, not silently assumed as settled.

Budget levels {2,4,8,16,32} per CLAUDE.md invariant #1. For SAMPLE,
budget B = number of samples in the majority vote (nested prefix). For
SELECT, the SAME token-equivalent budget (B * that problem's mean
completion length) is split between fewer real samples + PRM scoring
via budget/accounting.py's `budget_split_for_select` (already-tested,
not reimplemented here).

7 policies: Miser (A0/STOP always), Spendthrift (A1/SAMPLE always at
full B), UniformSelect (A2/SELECT always at full B), Gambler (random
STOP/SAMPLE, stop_probability=0.5, seeded), Oracle (per-budget ceiling:
correct if STOP, SAMPLE-at-B, or SELECT-at-B is correct), Detective
(post-hoc learned, real CV), Fortune Teller (pre-hoc learned, real CV).

H4: does Detective beat the best FIXED policy (Miser/Spendthrift/
UniformSelect/Gambler) at >=3/5 budget levels?
"""
import gc
import json
import sys
import time
import urllib.request
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, "src")

import numpy as np  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from marginal_token.answers.equivalence import check_equivalent  # noqa: E402
from marginal_token.answers.extraction import extract_answer  # noqa: E402
from marginal_token.answers.taxonomy import FailureStatus  # noqa: E402
from marginal_token.budget.accounting import (  # noqa: E402
    Charge, budget_split_for_select, charge_sample_action, charge_select_action,
)
from marginal_token.controller.base import Probe  # noqa: E402
from marginal_token.controller.features import FEATURE_NAMES, featurize  # noqa: E402
from marginal_token.controller.oracle_labels import _any_correct, _majority_correct  # noqa: E402
from marginal_token.pools.store import PoolStore  # noqa: E402
from marginal_token.scoring.pipeline import PRMScoreStore, compute_score_id  # noqa: E402
from marginal_token.selectors.prm_based import WeightedVoteEntry, prm_weighted_majority  # noqa: E402

POOL_ROOT = "results/pools"
SCORE_ROOT = "results/scores"
BUDGET_LEVELS = (2, 4, 8, 16, 32)
RANDOM_STATE = 20260826


def fetch_math500_all():
    found = {}
    offset = 0
    while offset < 500:
        url = (f"https://datasets-server.huggingface.co/rows?dataset=HuggingFaceH4/MATH-500"
               f"&config=default&split=test&offset={offset}&length=100")
        d = {}
        for _ in range(3):
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
            found[row["unique_id"]] = (row["problem"], row["answer"])
        offset += 100
    return found


def fetch_olympiad_all():
    found = {}
    offset = 0
    while offset < 674:
        url = (f"https://datasets-server.huggingface.co/rows?dataset=Hothan/OlympiadBench"
               f"&config=OE_TO_maths_en_COMP&split=train&offset={offset}&length=100")
        d = {}
        for _ in range(3):
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
            row_id = str(row["id"])
            if not row.get("is_multiple_answer") and not row.get("error") and row.get("final_answer"):
                found[row_id] = (row["question"], row["final_answer"][0])
        offset += 100
    return found


def select_correct_at_budget(ordered_samples, gold, n_select_samples):
    """PRM-weighted majority over the first n_select_samples (nested
    prefix), matching prm_based.py's real selector -- not a
    reimplementation of the voting rule, just wiring real data into it.
    """
    if n_select_samples <= 0:
        return False
    subset = ordered_samples[:n_select_samples]
    votes = []
    for s, score in subset:
        ext = extract_answer(s.text, finish_reason=s.finish_reason)
        if ext.status != FailureStatus.OK:
            votes.append(WeightedVoteEntry(answer_key=None, is_correct=None, weight=None))
            continue
        key = str(ext.value)
        eq = check_equivalent(prediction=ext.value, gold=gold)
        weight = score.mean_reward if (score is not None and score.status == "ok") else None
        votes.append(WeightedVoteEntry(answer_key=key, is_correct=bool(eq.equivalent), weight=weight))
    result = prm_weighted_majority(votes)
    return bool(result.is_correct)


def main():
    print("fetching MATH-500 + OlympiadBench-A (text + gold)...", flush=True)
    math_data = fetch_math500_all()
    oly_data = fetch_olympiad_all()
    print(f"got {len(math_data)} MATH-500, {len(oly_data)} OlympiadBench-A rows", flush=True)

    math_meta, oly_meta = [], []
    for jsonl_path in sorted(Path(POOL_ROOT).glob("*/*.jsonl")):
        pool_id = jsonl_path.parent.name
        problem_id = jsonl_path.stem.replace("__", "/")
        (oly_meta if problem_id.isdigit() else math_meta).append((pool_id, problem_id))

    pool_store = PoolStore(POOL_ROOT)
    score_store = PRMScoreStore(SCORE_ROOT)

    rows = []  # per-problem dict of everything needed downstream
    n_processed, n_no_scores = 0, 0

    def process(pool_id, pid, benchmark_id, text, gold):
        nonlocal n_no_scores
        pool = pool_store.load(pool_id, pid, benchmark_id, "qwen3.5-4b", "policy_primary")
        if len(pool) != 32:
            return None
        ordered_samples = sorted(pool.samples, key=lambda s: s.sample_idx)

        score_id = compute_score_id(pool_id, "primary_prm", "double_newline")
        try:
            scores = score_store.load(score_id, pid)
        except Exception:
            scores = []
        score_by_idx = {sc.sample_idx: sc for sc in scores}
        if not score_by_idx:
            n_no_scores += 1
        ordered_pairs = [(s, score_by_idx.get(s.sample_idx)) for s in ordered_samples]

        mean_len = sum(s.completion_tokens for s in ordered_samples) / len(ordered_samples)
        prm_forward_cost_tokens = mean_len  # disclosed assumption, see module docstring

        stop_correct = _majority_correct(ordered_samples[:4], gold)
        sample_correct = {b: _majority_correct(ordered_samples[:b], gold) for b in BUDGET_LEVELS}
        select_n = {b: budget_split_for_select(b * mean_len, mean_len, prm_forward_cost_tokens) for b in BUDGET_LEVELS}
        select_correct = {b: select_correct_at_budget(ordered_pairs, gold, select_n[b]) for b in BUDGET_LEVELS}

        probe = Probe(samples=ordered_samples[:4])
        feats = featurize(probe)

        return {
            "benchmark_id": benchmark_id, "mean_len": mean_len,
            "stop_correct": stop_correct, "sample_correct": sample_correct,
            "select_correct": select_correct, "select_n": select_n,
            "features": feats, "query_text": text,
        }

    for pool_id, pid in math_meta:
        if pid not in math_data:
            continue
        text, gold = math_data[pid]
        r = process(pool_id, pid, "math500", text, gold)
        if r is not None:
            rows.append(r)
        n_processed += 1
        if n_processed % 150 == 0:
            gc.collect()
            print(f"  {n_processed} processed", flush=True)

    for pool_id, pid in oly_meta:
        if pid not in oly_data:
            continue
        text, gold = oly_data[pid]
        r = process(pool_id, pid, "olympiad-a", text, gold)
        if r is not None:
            rows.append(r)
        n_processed += 1
        if n_processed % 150 == 0:
            gc.collect()
            print(f"  {n_processed} processed", flush=True)

    print(f"\ntotal usable problems: {len(rows)} (of {n_processed} scanned, {n_no_scores} had no PRM scores found)", flush=True)

    # ---- fit Detective + Fortune Teller under the same real CV as Day 14 ----
    benchmarks = np.array([r["benchmark_id"] for r in rows])
    X = np.array([[r["features"][f] for f in FEATURE_NAMES] for r in rows], dtype=float)
    texts = [r["query_text"] for r in rows]

    # oracle 4-class label per problem, for fitting Detective/FT (reuse the
    # already-computed correctness fields -- same STOP>SAMPLE>SELECT>ABSTAIN
    # priority as oracle_labels.oracle_action_label, but at the FULL N=32
    # budget specifically, matching how those models were trained tonight)
    def label_for(r):
        if r["stop_correct"]:
            return "stop"
        if r["sample_correct"][32]:
            return "sample"
        if r["select_correct"][32]:
            return "select"
        return "abstain"

    y = np.array([label_for(r) for r in rows])
    classes = sorted(set(y.tolist()))
    print(f"class distribution: {dict((c, int((y==c).sum())) for c in classes)}", flush=True)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    print("\nfitting Detective (post-hoc)...", flush=True)
    detective_pred = np.empty(len(y), dtype=object)
    for train_idx, test_idx in skf.split(X, benchmarks):
        X_train = X[train_idx]
        all_nan = np.isnan(X_train).all(axis=0)
        col_means = np.zeros(X_train.shape[1])
        if not all_nan.all():
            with np.errstate(invalid="ignore"):
                col_means[~all_nan] = np.nanmean(X_train[:, ~all_nan], axis=0)
        X_train_imp = np.where(np.isnan(X_train), col_means, X_train)
        X_test_imp = np.where(np.isnan(X[test_idx]), col_means, X[test_idx])
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_imp)
        X_test_scaled = scaler.transform(X_test_imp)
        model = LogisticRegression(max_iter=2000)
        model.fit(X_train_scaled, y[train_idx])
        detective_pred[test_idx] = model.predict(X_test_scaled)

    print("fitting Fortune Teller (pre-hoc)...", flush=True)
    from sentence_transformers import SentenceTransformer
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = np.asarray(embedder.encode(texts, show_progress_bar=False))
    ft_pred = np.empty(len(y), dtype=object)
    for train_idx, test_idx in skf.split(X, benchmarks):
        model = LogisticRegression(max_iter=2000)
        model.fit(embeddings[train_idx], y[train_idx])
        ft_pred[test_idx] = model.predict(embeddings[test_idx])

    # ---- E7: accuracy per policy per budget level ----
    def correct_for_action(r, action, b):
        if action == "stop":
            return r["stop_correct"]
        if action == "sample":
            return r["sample_correct"][b]
        if action == "select":
            return r["select_correct"][b]
        return False  # abstain

    def gambler_action(i, stop_p, rng):
        return "stop" if rng.random() < stop_p else "sample"

    import random
    gambler_rng = random.Random(0)

    policy_acc = defaultdict(dict)  # policy -> budget -> accuracy
    n = len(rows)
    for b in BUDGET_LEVELS:
        miser = sum(1 for r in rows if r["stop_correct"]) / n
        spendthrift = sum(1 for r in rows if r["sample_correct"][b]) / n
        uniform_select = sum(1 for r in rows if r["select_correct"][b]) / n
        oracle = sum(1 for r in rows if r["stop_correct"] or r["sample_correct"][b] or r["select_correct"][b]) / n
        gambler_rng.seed(0)
        gambler = sum(1 for r in rows if correct_for_action(r, gambler_action(0, 0.5, gambler_rng), b)) / n
        detective = sum(1 for r, a in zip(rows, detective_pred) if correct_for_action(r, a, b)) / n
        fortune_teller = sum(1 for r, a in zip(rows, ft_pred) if correct_for_action(r, a, b)) / n

        policy_acc["miser"][b] = miser
        policy_acc["spendthrift"][b] = spendthrift
        policy_acc["uniform_select"][b] = uniform_select
        policy_acc["gambler"][b] = gambler
        policy_acc["oracle"][b] = oracle
        policy_acc["detective"][b] = detective
        policy_acc["fortune_teller"][b] = fortune_teller

    print(f"\n=== E7: accuracy per policy per budget level (n={n} problems) ===")
    header = "policy".ljust(16) + "".join(f"B={b}".rjust(9) for b in BUDGET_LEVELS)
    print(header)
    for policy in ["miser", "spendthrift", "uniform_select", "gambler", "oracle", "fortune_teller", "detective"]:
        row_str = policy.ljust(16) + "".join(f"{policy_acc[policy][b]:.4f}".rjust(9) for b in BUDGET_LEVELS)
        print(row_str)

    print("\n=== H4: does Detective beat the best FIXED policy at each budget level? ===")
    fixed_policies = ["miser", "spendthrift", "uniform_select", "gambler"]
    wins = 0
    for b in BUDGET_LEVELS:
        best_fixed = max(policy_acc[p][b] for p in fixed_policies)
        det = policy_acc["detective"][b]
        beats = det > best_fixed
        wins += int(beats)
        print(f"  B={b}: Detective={det:.4f} vs best_fixed={best_fixed:.4f} -> {'BEATS' if beats else 'does not beat'}")
    print(f"\nH4 verdict: Detective beats best fixed policy at {wins}/5 budget levels "
          f"(need >=3/5 to accept) -> {'ACCEPT' if wins >= 3 else 'REJECT'}")

    # ---- E8: cost-per-correct-answer (token-equivalent) ----
    print(f"\n=== E8: mean token-equivalent cost per problem, at B=32 (PRM forward assumed = 1 sample's cost) ===")
    prm_forward_cost_tokens_global = sum(r["mean_len"] for r in rows) / n  # rough global figure for reporting
    for policy in ["miser", "spendthrift", "uniform_select"]:
        total_cost, total_correct = 0.0, 0
        for r in rows:
            b = 32
            if policy == "miser":
                charge = Charge(policy_tokens=0)
                correct = r["stop_correct"]
            elif policy == "spendthrift":
                charge = charge_sample_action(n_samples=b, tokens_per_sample=int(r["mean_len"]))
                correct = r["sample_correct"][b]
            else:
                charge = charge_select_action(budget_b=int(b * r["mean_len"]), tokens_per_sample=int(r["mean_len"]),
                                                prm_forward_cost_tokens=r["mean_len"])
                correct = r["select_correct"][b]
            total_cost += charge.total_token_equivalent(prm_forward_cost_tokens=r["mean_len"])
            total_correct += int(correct)
        cost_per_correct = total_cost / max(total_correct, 1)
        print(f"  {policy}: mean_cost={total_cost/n:.1f} tokens/problem, accuracy={total_correct/n:.4f}, "
              f"cost_per_correct_answer={cost_per_correct:.1f} tokens")

    print("\n(E8 note: SELECT's cost-per-correct here inherits the disclosed prm_forward_cost_tokens=mean_len "
          "assumption -- a real, sourced PRM cost figure would change this number, not the qualitative story "
          "given SELECT's already-confirmed near-zero win rate.)")


if __name__ == "__main__":
    main()
