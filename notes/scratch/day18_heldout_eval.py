"""
Day 18: THE held-out evaluation. One pass, no re-tuning, per invariant
#8 and the heldout configs' own `rule` field -- whatever this prints is
final.

Loads the FROZEN Detective model (results/models/detective_frozen.joblib,
fit last night on the dev set only, Day 17) -- does NOT refit on
held-out data, which would be exactly the re-tuning this gate forbids.
Runs it on P4 (OlympiadBench-B, in-distribution) and P5 (AIME25,
out-of-distribution) separately, matching how the configs themselves
frame the distinction (heldout_in_distribution vs.
heldout_out_of_distribution) -- reporting them pooled would blur a
distinction the project's own design cares about.

P5 (n=30) is below any reasonable CI floor and its own config says
`statistics: ordinal_only` -- medians/counts only for that one, no
means dressed up with false precision (same honesty rule
evaluation/stats.py enforces elsewhere).

Canonical-only pool enumeration throughout (compute_pool_id, never a
glob) -- same discipline as every held-out script tonight.
"""
import json
import sys
import urllib.request

sys.path.insert(0, "src")

import joblib  # noqa: E402
import numpy as np  # noqa: E402

from marginal_token.answers.extraction import extract_answer  # noqa: E402
from marginal_token.answers.taxonomy import FailureStatus  # noqa: E402
from marginal_token.controller.base import Budget, Probe  # noqa: E402
from marginal_token.controller.features import featurize  # noqa: E402
from marginal_token.controller.oracle_labels import _majority_correct, _any_correct, oracle_action_label  # noqa: E402
from marginal_token.pools.store import PoolStore, compute_pool_id  # noqa: E402
from marginal_token.generation.run_sweeps import fetch_olympiad_bench_problems, fetch_aime25_problems  # noqa: E402


def fetch_olympiad_gold(ids_file):
    wanted = set(str(x) for x in json.load(open(ids_file)))
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


def fetch_aime25_gold(ids_file):
    wanted = set(json.load(open(ids_file)))
    found = {}
    rev = "a6ad95f611d72cf628a80b58bd0432ef6638f958"
    for config, fname in [("AIME2025-I", "aime2025-I.jsonl"), ("AIME2025-II", "aime2025-II.jsonl")]:
        url = f"https://huggingface.co/datasets/opencompass/AIME2025/resolve/{rev}/{fname}"
        text = urllib.request.urlopen(url, timeout=30).read().decode("utf-8")
        for i, line in enumerate(text.strip().split("\n")):
            row = json.loads(line)
            pid = f"{config}-{i:02d}"
            if pid in wanted:
                found[pid] = row["answer"]
    return found


def eval_benchmark(name, benchmark_id, ids, gold, controller):
    store = PoolStore("results/pools")
    n = len(ids)
    stop_correct, sample_correct, select_correct = [], [], []
    detective_correct, detective_actions = [], []

    for pid in ids:
        if pid not in gold:
            continue
        pool_id = compute_pool_id(policy_ref="qwen3.5-4b", backend_ref="policy_primary", benchmark_id=benchmark_id,
                                    problem_id=pid, temperature=0.8, top_p=0.95, max_tokens=1024, seed=None, n=32)
        pool = store.load(pool_id, pid, benchmark_id, "qwen3.5-4b", "policy_primary")
        assert len(pool) == 32, f"{pid}: expected 32 samples, got {len(pool)}"
        ordered = sorted(pool.samples, key=lambda s: s.sample_idx)

        g = gold[pid]
        sc = _majority_correct(ordered[:4], g)
        smc = _majority_correct(ordered, g)
        slc = _any_correct(ordered, g)
        stop_correct.append(sc)
        sample_correct.append(smc)
        select_correct.append(slc)

        probe = Probe(samples=ordered[:4])
        feats = featurize(probe)
        decision = controller.decide(probe, Budget(max_tokens=32, max_latency_ms=None))
        detective_actions.append(decision.action)
        if decision.action == "stop":
            detective_correct.append(sc)
        elif decision.action == "sample":
            detective_correct.append(smc)
        elif decision.action == "select":
            detective_correct.append(slc)
        else:
            detective_correct.append(False)

    n_eval = len(stop_correct)
    stop_arr = np.array(stop_correct)
    sample_arr = np.array(sample_correct)
    select_arr = np.array(select_correct)
    oracle_arr = stop_arr | sample_arr | select_arr
    best_fixed = max(stop_arr.mean(), sample_arr.mean())
    gap = 100 * (oracle_arr.mean() - best_fixed)
    detective_acc = np.mean(detective_correct)

    print(f"\n{'='*70}\n{name} (n={n_eval}/{n})\n{'='*70}")
    print(f"STOP accuracy:          {stop_arr.mean():.4f}")
    print(f"SAMPLE accuracy:        {sample_arr.mean():.4f}")
    print(f"SELECT ceiling:         {select_arr.mean():.4f}")
    print(f"Oracle ceiling:         {oracle_arr.mean():.4f}")
    print(f"Best fixed policy:      {best_fixed:.4f}")
    print(f"G1-style gap:           {gap:.2f}pp")
    print(f"Detective (frozen model) achieved accuracy: {detective_acc:.4f}")
    print(f"Detective vs best fixed: {100*(detective_acc - best_fixed):+.2f}pp")
    from collections import Counter
    print(f"Detective action distribution: {dict(Counter(detective_actions))}")
    from collections import Counter as C2
    oracle_actions = []
    for i in range(n_eval):
        if stop_arr[i]:
            oracle_actions.append("stop")
        elif sample_arr[i]:
            oracle_actions.append("sample")
        elif select_arr[i]:
            oracle_actions.append("select")
        else:
            oracle_actions.append("abstain")
    print(f"TRUE oracle action distribution: {dict(C2(oracle_actions))}")

    if n_eval < 15:
        print(f"\n(n={n_eval} -- ordinal reporting only per this benchmark's own config, no CI. "
              f"Raw correct/total counts: STOP {int(stop_arr.sum())}/{n_eval}, "
              f"SAMPLE {int(sample_arr.sum())}/{n_eval}, oracle {int(oracle_arr.sum())}/{n_eval}, "
              f"Detective {int(sum(detective_correct))}/{n_eval}.)")

    return {
        "n": n_eval, "stop": float(stop_arr.mean()), "sample": float(sample_arr.mean()),
        "select_ceiling": float(select_arr.mean()), "oracle": float(oracle_arr.mean()),
        "best_fixed": float(best_fixed), "gap_pp": float(gap), "detective": float(detective_acc),
    }


def main():
    controller = joblib.load("results/models/detective_frozen.joblib")
    print("loaded FROZEN Detective model (results/models/detective_frozen.joblib, fit on dev set only)")

    p4_ids = [str(x) for x in json.load(open("configs/benchmarks/data/heldout-olympiad-b-ids.json"))]
    p4_gold = fetch_olympiad_gold("configs/benchmarks/data/heldout-olympiad-b-ids.json")
    p4_result = eval_benchmark("P4: OlympiadBench-B (held-out, IN-distribution)", "olympiad-b-heldout", p4_ids, p4_gold, controller)

    p5_ids = json.load(open("configs/benchmarks/data/heldout-aime25-ids.json"))
    p5_gold = fetch_aime25_gold("configs/benchmarks/data/heldout-aime25-ids.json")
    p5_result = eval_benchmark("P5: AIME25 (held-out, OUT-OF-distribution)", "aime25-heldout", p5_ids, p5_gold, controller)

    print(f"\n{'='*70}\nFINAL HELD-OUT RESULT -- recorded as-is, no re-tuning\n{'='*70}")
    print(json.dumps({"P4_olympiad_b": p4_result, "P5_aime25": p5_result}, indent=2))

    with open("results/heldout_results.json", "w") as f:
        json.dump({"P4_olympiad_b": p4_result, "P5_aime25": p5_result}, f, indent=2)
    print("\nwrote results/heldout_results.json")


if __name__ == "__main__":
    main()
