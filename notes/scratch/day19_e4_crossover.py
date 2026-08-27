"""
Day 13's E4, done properly now (Day 19): PRM-argmax crossover N* with
CI, and the length/step-count regression that verifies V4 ("PRM argmax
winners are length-biased," docs/brief.md line 115, citing 2606.09078).

Pure analysis over already-cached data (P1+P2 pools + PRM scores) --
no new generation. Canonical-only pool enumeration throughout (same
discipline as every script since the duplicate-pool bug was found).

Crossover N*: at each budget level N in {2,4,8,16,32} (nested prefix),
compare PRM-argmax accuracy to plain-majority accuracy, paired per
problem. If PRM-argmax ever becomes reliably better (BCa 95% CI on the
mean paired difference excludes zero, positive), N* is the smallest
such N. Per docs/brief.md line 824's own honest fallback: "no crossover
observed up to N=32" is a legitimate, expected-given-everything-else-
found-tonight result, not a failure to find one.

Length/step bias (V4): for PRM-argmax's selected winner at N=32, is its
completion length/step count systematically different from the pool's
median? A simple paired comparison (winner vs. pool median), not a
full regression model -- matches what the brief actually asks for
("regress winner length and step count against pool median").
"""
import gc
import json
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, "src")

import numpy as np  # noqa: E402

from marginal_token.answers.equivalence import check_equivalent  # noqa: E402
from marginal_token.answers.extraction import extract_answer  # noqa: E402
from marginal_token.answers.taxonomy import FailureStatus  # noqa: E402
from marginal_token.evaluation.stats import paired_bootstrap_bca  # noqa: E402
from marginal_token.pools.store import PoolStore, compute_pool_id  # noqa: E402
from marginal_token.scoring.pipeline import PRMScoreStore, compute_score_id  # noqa: E402
from marginal_token.scoring.segmentation import segment_double_newline  # noqa: E402
from marginal_token.selectors.basic import VoteEntry, plain_majority  # noqa: E402
from marginal_token.selectors.prm_based import WeightedVoteEntry, prm_argmax  # noqa: E402

BUDGET_LEVELS = (2, 4, 8, 16, 32)


def fetch_math500_gold_all():
    found = {}
    offset = 0
    while offset < 500:
        url = (f"https://datasets-server.huggingface.co/rows?dataset=HuggingFaceH4/MATH-500"
               f"&config=default&split=test&offset={offset}&length=100")
        d = json.load(urllib.request.urlopen(url, timeout=30))
        for r in d["rows"]:
            row = r["row"]
            found[row["unique_id"]] = row["answer"]
        offset += 100
    return found


def fetch_olympiad_gold_all(wanted_ids):
    wanted = set(wanted_ids)
    found = {}
    offset = 0
    while offset < 674 and len(found) < len(wanted):
        url = (f"https://datasets-server.huggingface.co/rows?dataset=Hothan/OlympiadBench"
               f"&config=OE_TO_maths_en_COMP&split=train&offset={offset}&length=100")
        d = json.load(urllib.request.urlopen(url, timeout=30))
        for r in d["rows"]:
            row = r["row"]
            row_id = str(row["id"])
            if row_id in wanted and not row.get("is_multiple_answer") and not row.get("error") and row.get("final_answer"):
                found[row_id] = row["final_answer"][0]
        offset += 100
    return found


def main():
    print("fetching gold answers...", flush=True)
    math_gold = fetch_math500_gold_all()
    oly_ids = [str(x) for x in json.load(open("configs/benchmarks/data/olympiad-a-ids.json"))]
    oly_gold = fetch_olympiad_gold_all(oly_ids)
    print(f"got {len(math_gold)} MATH-500, {len(oly_gold)}/{len(oly_ids)} OlympiadBench-A", flush=True)

    pool_store = PoolStore("results/pools")
    score_store = PRMScoreStore("results/scores")

    majority_correct_by_n = {n: [] for n in BUDGET_LEVELS}
    argmax_correct_by_n = {n: [] for n in BUDGET_LEVELS}
    winner_length_deltas, winner_step_deltas = [], []
    n_processed, n_no_scores = 0, 0

    def process(pid, benchmark_id, gold):
        nonlocal n_no_scores
        pool_id = compute_pool_id(policy_ref="qwen3.5-4b", backend_ref="policy_primary", benchmark_id=benchmark_id,
                                    problem_id=pid, temperature=0.8, top_p=0.95, max_tokens=1024, seed=None, n=32)
        pool = pool_store.load(pool_id, pid, benchmark_id, "qwen3.5-4b", "policy_primary")
        if len(pool) != 32:
            return
        ordered = sorted(pool.samples, key=lambda s: s.sample_idx)

        score_id = compute_score_id(pool_id, "primary_prm", "double_newline")
        try:
            scores = {sc.sample_idx: sc for sc in score_store.load(score_id, pid)}
        except Exception:
            scores = {}
        if not scores:
            n_no_scores += 1
            return

        lengths = [s.completion_tokens for s in ordered]
        steps = [len(segment_double_newline(s.text)) for s in ordered]
        median_len, median_step = float(np.median(lengths)), float(np.median(steps))

        for n in BUDGET_LEVELS:
            subset = ordered[:n]
            plain_votes = []
            weighted_votes = []
            for s in subset:
                ext = extract_answer(s.text, finish_reason=s.finish_reason)
                key = str(ext.value) if ext.status == FailureStatus.OK else None
                is_correct = check_equivalent(prediction=ext.value, gold=gold).equivalent if key is not None else None
                plain_votes.append(VoteEntry(answer_key=key, is_correct=is_correct))
                sc = scores.get(s.sample_idx)
                weight = sc.mean_reward if (sc is not None and sc.status == "ok") else None
                weighted_votes.append(WeightedVoteEntry(answer_key=key, is_correct=is_correct, weight=weight))

            maj = plain_majority(plain_votes)
            am = prm_argmax(weighted_votes)
            majority_correct_by_n[n].append(bool(maj.is_correct))
            argmax_correct_by_n[n].append(bool(am.is_correct))

            if n == 32 and am.winning_key is not None:
                winner_idx = None
                for i, s in enumerate(subset):
                    ext = extract_answer(s.text, finish_reason=s.finish_reason)
                    if ext.status == FailureStatus.OK and str(ext.value) == am.winning_key:
                        sc = scores.get(s.sample_idx)
                        if sc is not None and sc.mean_reward == max(
                            (scores[s2.sample_idx].mean_reward for s2 in subset
                             if scores.get(s2.sample_idx) and scores[s2.sample_idx].status == "ok"),
                            default=None,
                        ):
                            winner_idx = i
                            break
                if winner_idx is not None:
                    winner_length_deltas.append(lengths[winner_idx] - median_len)
                    winner_step_deltas.append(steps[winner_idx] - median_step)

    for pid, gold in math_gold.items():
        process(pid, "math500", gold)
        n_processed += 1
        if n_processed % 150 == 0:
            gc.collect()
            print(f"  {n_processed} processed", flush=True)

    for pid in oly_ids:
        if pid not in oly_gold:
            continue
        process(pid, "olympiad-a", oly_gold[pid])
        n_processed += 1
        if n_processed % 150 == 0:
            gc.collect()
            print(f"  {n_processed} processed", flush=True)

    print(f"\ntotal usable problems (with PRM scores): {len(majority_correct_by_n[32])} "
          f"(of {n_processed} scanned, {n_no_scores} had no PRM scores)", flush=True)

    print(f"\n=== Crossover analysis: PRM-argmax vs plain majority, by budget level N ===")
    print(f"{'N':>4}{'majority_acc':>14}{'argmax_acc':>12}{'diff':>10}{'CI_lo':>10}{'CI_hi':>10}{'excl_zero':>11}")
    crossover_n = None
    for n in BUDGET_LEVELS:
        maj_arr = np.array(majority_correct_by_n[n], dtype=float)
        am_arr = np.array(argmax_correct_by_n[n], dtype=float)
        diff = am_arr - maj_arr
        boot = paired_bootstrap_bca(diff, seed=20260827)
        excludes_zero = boot.ci_lo > 0
        if excludes_zero and crossover_n is None:
            crossover_n = n
        print(f"{n:>4}{maj_arr.mean():>14.4f}{am_arr.mean():>12.4f}{diff.mean():>10.4f}"
              f"{boot.ci_lo:>10.4f}{boot.ci_hi:>10.4f}{str(excludes_zero):>11}")

    if crossover_n is not None:
        print(f"\nCrossover N* = {crossover_n} (smallest N where PRM-argmax's CI excludes zero, positive)")
    else:
        print(f"\nNo crossover observed up to N=32 -- PRM-argmax never reliably beats plain majority "
              f"at any tested budget level. Per docs/brief.md line 824, this is the honest report, "
              f"not a failure to find a crossover.")

    print(f"\n=== V4: are PRM-argmax winners (at N=32) length/step-biased vs. the pool median? ===")
    len_arr = np.array(winner_length_deltas)
    step_arr = np.array(winner_step_deltas)
    print(f"n winners identified: {len(len_arr)}")
    if len(len_arr) >= 5:
        len_boot = paired_bootstrap_bca(len_arr, seed=20260827)
        step_boot = paired_bootstrap_bca(step_arr, seed=20260827)
        print(f"length delta (winner - pool median): mean={len_arr.mean():.2f} tokens, "
              f"BCa 95% CI [{len_boot.ci_lo:.2f}, {len_boot.ci_hi:.2f}], excludes zero: {len_boot.ci_lo > 0 or len_boot.ci_hi < 0}")
        print(f"step delta (winner - pool median): mean={step_arr.mean():.2f} steps, "
              f"BCa 95% CI [{step_boot.ci_lo:.2f}, {step_boot.ci_hi:.2f}], excludes zero: {step_boot.ci_lo > 0 or step_boot.ci_hi < 0}")
        print(f"\nV4 verdict: {'CONFIRMED -- PRM-argmax winners are length-biased' if (len_boot.ci_lo > 0 or len_boot.ci_hi < 0) else 'NOT CONFIRMED -- no reliable length bias detected in this data'}")
    else:
        print("too few winners identified for a CI -- reporting raw values only")
        print(f"length deltas: {len_arr.tolist()}")
        print(f"step deltas: {step_arr.tolist()}")


if __name__ == "__main__":
    main()
