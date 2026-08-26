"""
Day 11: E0 thin end-to-end slice + Gate G9. Full Diagnose -> Predict ->
Allocate chain on the frozen 100-problem dev-100 slice (already
generated + scored as part of P1) -- preliminary H1/H2/H4 numbers, not
final (Day 12-15 do those for real at full scale / with proper CV).
"""

import json
import sys
import urllib.request
import time
from pathlib import Path

sys.path.insert(0, "src")

import numpy as np

from marginal_token.controller.base import Probe  # noqa: E402
from marginal_token.controller.features import featurize  # noqa: E402
from marginal_token.controller.oracle_labels import oracle_action_label  # noqa: E402
from marginal_token.controller.predictor import DetectiveController  # noqa: E402
from marginal_token.answers.extraction import extract_answer  # noqa: E402
from marginal_token.answers.equivalence import check_equivalent  # noqa: E402
from marginal_token.answers.taxonomy import FailureStatus  # noqa: E402
from marginal_token.pools.store import PoolStore  # noqa: E402

POOL_ROOT = "results/pools"
IDS_PATH = "configs/benchmarks/data/math500-dev100-ids.json"
BUDGET_LEVELS = [2, 4, 8, 16, 32]


def fetch_gold(ids):
    wanted = set(ids)
    found = {}
    offset = 0
    while offset < 500 and len(found) < len(wanted):
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
            if row["unique_id"] in wanted:
                found[row["unique_id"]] = row["answer"]
        offset += 100
    return found


def majority_correct(samples, gold):
    from collections import Counter
    counts, correctness = Counter(), {}
    for s in samples:
        ext = extract_answer(s.text, finish_reason=s.finish_reason)
        if ext.status != FailureStatus.OK:
            continue
        key = str(ext.value)
        counts[key] += 1
        if key not in correctness:
            eq = check_equivalent(prediction=ext.value, gold=gold)
            correctness[key] = bool(eq.equivalent)
    if not counts:
        return False
    top_key, top_count = counts.most_common(1)[0]
    if sum(1 for c in counts.values() if c == top_count) > 1:
        return False
    return correctness[top_key]


def main():
    ids = json.load(open(IDS_PATH))
    assert len(ids) == 100
    gold = fetch_gold(ids)
    print(f"fetched {len(gold)} gold answers for the 100-problem dev slice")

    store = PoolStore(POOL_ROOT)
    pool_id_by_pid = {}
    for jsonl_path in Path(POOL_ROOT).glob("*/*.jsonl"):
        pid = jsonl_path.stem.replace("__", "/")
        if pid in gold:
            pool_id_by_pid[pid] = jsonl_path.parent.name

    pools = {}
    for pid, pool_id in pool_id_by_pid.items():
        pool = store.load(pool_id, pid, "math500", "qwen3.5-4b", "policy_primary")
        assert len(pool) == 32
        pools[pid] = pool

    # === DIAGNOSE: G1 on this exact 100-problem slice ===
    stop_arr, sample_arr, select_arr, labels = [], [], [], []
    features_list, y = [], []
    for pid, pool in pools.items():
        result = oracle_action_label(pool.samples, gold[pid])
        stop_arr.append(result.stop_correct)
        sample_arr.append(result.sample_correct)
        select_arr.append(result.select_correct)
        labels.append(result.action)
        probe = sorted(pool.samples, key=lambda s: s.sample_idx)[:4]
        features_list.append(featurize(Probe(samples=probe)))
        y.append(result.action)

    stop_arr, sample_arr, select_arr = np.array(stop_arr), np.array(sample_arr), np.array(select_arr)
    oracle_acc = (stop_arr | sample_arr | select_arr).mean()
    best_fixed = max(stop_arr.mean(), sample_arr.mean())
    print(f"\n=== DIAGNOSE (H1), 100-problem slice ===")
    print(f"STOP={stop_arr.mean():.3f} SAMPLE={sample_arr.mean():.3f} SELECT_ceiling={select_arr.mean():.3f} "
          f"oracle={oracle_acc:.3f} best_fixed={best_fixed:.3f} GAP={100*(oracle_acc-best_fixed):.2f}pp")

    # === PREDICT: Detective in-sample fit (H2/H3 preliminary) ===
    detective = DetectiveController()
    detective.fit(features_list, y)
    from marginal_token.controller.features import FEATURE_NAMES
    arr = np.array([[f[n] for n in FEATURE_NAMES] for f in features_list], dtype=float)
    for i, name in enumerate(FEATURE_NAMES):
        col = arr[:, i]
        mask = np.isnan(col)
        if mask.any():
            arr[mask, i] = detective._impute_values[name]
    arr_scaled = detective.scaler.transform(arr)
    preds = detective.model.predict(arr_scaled)
    detective_acc = float(np.mean(preds == np.array(y)))
    print(f"\n=== PREDICT (H2/H3 preliminary, in-sample), 100-problem slice ===")
    from collections import Counter as C
    majority_class_acc = max(C(y).values()) / len(y)
    print(f"majority-class floor: {majority_class_acc:.3f}, Detective in-sample: {detective_acc:.3f}")

    # === ALLOCATE (H4 preliminary): Detective-driven vs fixed policies at 5 budget levels ===
    print(f"\n=== ALLOCATE (H4 preliminary), 100-problem slice, 5 budget levels ===")
    print(f"{'k':>4}{'miser(stop)':>14}{'spendthrift(sample@k)':>24}{'detective-driven':>18}")
    beats_count = 0
    for k in BUDGET_LEVELS:
        miser_correct, spend_correct, det_correct = [], [], []
        for pid, pool, feats, action in zip(pools.keys(), pools.values(), features_list, y):
            probe = sorted(pool.samples, key=lambda s: s.sample_idx)[:4]
            prefix_k = sorted(pool.samples, key=lambda s: s.sample_idx)[:k]
            miser_correct.append(majority_correct(probe, gold[pid]))
            spend_correct.append(majority_correct(prefix_k, gold[pid]))

            row = np.array([[feats[n] if not np.isnan(feats[n]) else detective._impute_values[n]
                              for n in FEATURE_NAMES]])
            row_scaled = detective.scaler.transform(row)
            pred_action = detective.model.predict(row_scaled)[0]
            if pred_action == "stop":
                det_correct.append(majority_correct(probe, gold[pid]))
            elif pred_action == "sample":
                det_correct.append(majority_correct(prefix_k, gold[pid]))
            else:  # abstain / select -- select never predicted (not in training labels beyond rare cases)
                det_correct.append(False)

        m, s, d = np.mean(miser_correct), np.mean(spend_correct), np.mean(det_correct)
        beats_both = d > m and d > s
        if beats_both:
            beats_count += 1
        print(f"{k:>4}{m:>14.3f}{s:>24.3f}{d:>18.3f}{'  <- beats both fixed' if beats_both else ''}")

    print(f"\nDetective beats BOTH fixed policies at {beats_count}/5 budget levels "
          f"(H4 needs >=3/5 beating EVERY fixed policy -- this in-sample check is optimistic, "
          f"real evaluation is Day 15's job).")

    print("\n(Preliminary, 100-problem slice, in-sample Detective fit -- NOT final. "
          "Day 12-15 do real grouped-CV predictor fitting and full-scale H1/H2/H3/H4 evaluation.)")

    # === Day 11's literal "done when": one figure per H1/H2/H4, even at slice scale ===
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path("results/figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    # H1: STOP / SAMPLE / SELECT-ceiling / oracle vs best-fixed
    fig, ax = plt.subplots(figsize=(6, 4))
    labels_h1 = ["STOP", "SAMPLE", "SELECT\nceiling", "Oracle"]
    vals_h1 = [stop_arr.mean(), sample_arr.mean(), select_arr.mean(), oracle_acc]
    bars = ax.bar(labels_h1, vals_h1, color=["#4C72B0", "#55A868", "#8172B2", "#C44E52"])
    for b, v in zip(bars, vals_h1):
        ax.text(b.get_x() + b.get_width()/2, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)
    ax.axhline(best_fixed, color="gray", linestyle="--", linewidth=1, label=f"best fixed = {best_fixed:.3f}")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("accuracy")
    ax.set_title(f"E0 (Day 11): H1 diagnose, 100-problem slice\nGAP = {100*(oracle_acc-best_fixed):.2f}pp")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "day11_e0_h1_diagnose.png", dpi=150)

    # H2: predictor accuracy vs majority-class floor
    fig, ax = plt.subplots(figsize=(5, 4))
    labels_h2 = ["majority\nclass", "Detective\n(in-sample)"]
    vals_h2 = [majority_class_acc, detective_acc]
    bars = ax.bar(labels_h2, vals_h2, color=["#4C72B0", "#55A868"])
    for b, v in zip(bars, vals_h2):
        ax.text(b.get_x() + b.get_width()/2, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("action-label accuracy")
    ax.set_title("E0 (Day 11): H2 predict (preliminary, in-sample), 100-problem slice")
    fig.tight_layout()
    fig.savefig(out_dir / "day11_e0_h2_predict.png", dpi=150)

    # H4: accuracy vs budget k for miser / spendthrift / detective-driven
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(BUDGET_LEVELS, [0.680]*5, marker="o", label="Miser (always STOP)")
    # recompute the three curves cleanly for plotting (avoid relying on the loop-local vars above)
    miser_curve, spend_curve, det_curve = [], [], []
    for k in BUDGET_LEVELS:
        miser_correct, spend_correct, det_correct = [], [], []
        for pid, pool, feats, action in zip(pools.keys(), pools.values(), features_list, y):
            probe = sorted(pool.samples, key=lambda s: s.sample_idx)[:4]
            prefix_k = sorted(pool.samples, key=lambda s: s.sample_idx)[:k]
            miser_correct.append(majority_correct(probe, gold[pid]))
            spend_correct.append(majority_correct(prefix_k, gold[pid]))
            row = np.array([[feats[n] if not np.isnan(feats[n]) else detective._impute_values[n]
                              for n in FEATURE_NAMES]])
            row_scaled = detective.scaler.transform(row)
            pred_action = detective.model.predict(row_scaled)[0]
            if pred_action == "stop":
                det_correct.append(majority_correct(probe, gold[pid]))
            elif pred_action == "sample":
                det_correct.append(majority_correct(prefix_k, gold[pid]))
            else:
                det_correct.append(False)
        miser_curve.append(np.mean(miser_correct))
        spend_curve.append(np.mean(spend_correct))
        det_curve.append(np.mean(det_correct))
    ax.clear()
    ax.plot(BUDGET_LEVELS, miser_curve, marker="o", label="Miser (always STOP)")
    ax.plot(BUDGET_LEVELS, spend_curve, marker="s", label="Spendthrift (always SAMPLE@k)")
    ax.plot(BUDGET_LEVELS, det_curve, marker="^", label="Detective-driven")
    ax.set_xscale("log", base=2)
    ax.set_xticks(BUDGET_LEVELS)
    ax.set_xticklabels(BUDGET_LEVELS)
    ax.set_xlabel("budget k (samples)")
    ax.set_ylabel("accuracy")
    ax.set_ylim(0, 1.0)
    ax.set_title("E0 (Day 11): H4 allocate (preliminary), 100-problem slice")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "day11_e0_h4_allocate.png", dpi=150)

    print(f"\n3 figures saved to {out_dir}: day11_e0_h1_diagnose.png, day11_e0_h2_predict.png, "
          f"day11_e0_h4_allocate.png -- Day 11's literal 'done when' condition.")


if __name__ == "__main__":
    main()
