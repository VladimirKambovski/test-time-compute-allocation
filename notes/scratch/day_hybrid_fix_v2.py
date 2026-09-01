"""
Disclosed second attempt at the hybrid-fallback fix (Fix #2), after the
first attempt's selection criterion (max raw dev-CV accuracy) trivially
picked a threshold that never triggered. Corrected criterion, decided
before touching held-out again: best sample-recall gain within a small
accuracy-cost tolerance (mirrors Fix #1's own selection logic) --
threshold=0.6 (dev CV: accuracy 0.9244->0.9111, -1.33pp; sample_recall
0.1273->0.2182, +9pp). This is the ONE disclosed second held-out touch
for this specific idea, not a search -- threshold=0.6 was never
evaluated against held-out in the first attempt.
"""
import json
import sys
import time
import urllib.request

sys.path.insert(0, "src")

import numpy as np  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from marginal_token.controller.base import Probe  # noqa: E402
from marginal_token.controller.features import FEATURE_NAMES, featurize  # noqa: E402
from marginal_token.controller.oracle_labels import oracle_action_label, _majority_correct, _any_correct  # noqa: E402
from marginal_token.pools.store import PoolStore, compute_pool_id  # noqa: E402


def _fetch_json_retry(url, attempts=5):
    for i in range(attempts):
        try:
            return json.load(urllib.request.urlopen(url, timeout=30))
        except Exception:
            if i == attempts - 1:
                raise
            time.sleep(3)


def fetch_math500_all():
    found = {}
    offset = 0
    while offset < 500:
        url = (f"https://datasets-server.huggingface.co/rows?dataset=HuggingFaceH4/MATH-500"
               f"&config=default&split=test&offset={offset}&length=100")
        d = _fetch_json_retry(url)
        for r in d["rows"]:
            row = r["row"]
            found[row["unique_id"]] = row["answer"]
        offset += 100
    return found


def fetch_olympiad_all(wanted_ids):
    wanted = set(wanted_ids)
    found = {}
    offset = 0
    while offset < 674 and len(found) < len(wanted):
        url = (f"https://datasets-server.huggingface.co/rows?dataset=Hothan/OlympiadBench"
               f"&config=OE_TO_maths_en_COMP&split=train&offset={offset}&length=100")
        d = _fetch_json_retry(url)
        for r in d["rows"]:
            row = r["row"]
            row_id = str(row["id"])
            if row_id in wanted and not row.get("is_multiple_answer") and not row.get("error") and row.get("final_answer"):
                found[row_id] = row["final_answer"][0]
        offset += 100
    return found


def build_dev_data():
    math_gold = fetch_math500_all()
    oly_ids = [str(x) for x in json.load(open("configs/benchmarks/data/olympiad-a-ids.json"))]
    oly_gold = fetch_olympiad_all(oly_ids)

    store = PoolStore("results/pools")
    X_dicts, y = [], []

    for pid, gold in math_gold.items():
        pool_id = compute_pool_id(policy_ref="qwen3.5-4b", backend_ref="policy_primary", benchmark_id="math500",
                                    problem_id=pid, temperature=0.8, top_p=0.95, max_tokens=1024, seed=None, n=32)
        pool = store.load(pool_id, pid, "math500", "qwen3.5-4b", "policy_primary")
        assert len(pool) == 32
        ordered = sorted(pool.samples, key=lambda s: s.sample_idx)
        X_dicts.append(featurize(Probe(samples=ordered[:4])))
        y.append(oracle_action_label(pool.samples, gold).action)

    for pid in oly_ids:
        if pid not in oly_gold:
            continue
        gold = oly_gold[pid]
        pool_id = compute_pool_id(policy_ref="qwen3.5-4b", backend_ref="policy_primary", benchmark_id="olympiad-a",
                                    problem_id=pid, temperature=0.8, top_p=0.95, max_tokens=1024, seed=None, n=32)
        pool = store.load(pool_id, pid, "olympiad-a", "qwen3.5-4b", "policy_primary")
        assert len(pool) == 32
        ordered = sorted(pool.samples, key=lambda s: s.sample_idx)
        X_dicts.append(featurize(Probe(samples=ordered[:4])))
        y.append(oracle_action_label(pool.samples, gold).action)

    X = np.array([[d[f] for f in FEATURE_NAMES] for d in X_dicts], dtype=float)
    y = np.array(y)
    return X, y


def main():
    print("rebuilding dev features/labels (canonical enumeration)...", flush=True)
    X, y = build_dev_data()
    print(f"n={len(y)}", flush=True)

    all_nan = np.isnan(X).all(axis=0)
    col_means = np.zeros(X.shape[1])
    if not all_nan.all():
        with np.errstate(invalid="ignore"):
            col_means[~all_nan] = np.nanmean(X[:, ~all_nan], axis=0)
    X_imp = np.where(np.isnan(X), col_means, X)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imp)
    plain_model = LogisticRegression(max_iter=2000)
    plain_model.fit(X_scaled, y.tolist())

    def predict_action(feats_dict, threshold):
        row = np.array([[feats_dict[f] if not np.isnan(feats_dict[f]) else col_means[i]
                          for i, f in enumerate(FEATURE_NAMES)]])
        row_scaled = scaler.transform(row)
        probs = plain_model.predict_proba(row_scaled)[0]
        class_probs = dict(zip(plain_model.classes_, probs))
        action = max(class_probs, key=class_probs.get)
        if max(class_probs.values()) < threshold:
            action = "sample"
        return action

    p4_ids = [str(x) for x in json.load(open("configs/benchmarks/data/heldout-olympiad-b-ids.json"))]
    p4_gold = fetch_olympiad_all(p4_ids)
    store = PoolStore("results/pools")

    n, correct = 0, 0
    actions_used = {}
    for pid in p4_ids:
        if pid not in p4_gold:
            continue
        gold = p4_gold[pid]
        pool_id = compute_pool_id(policy_ref="qwen3.5-4b", backend_ref="policy_primary", benchmark_id="olympiad-b-heldout",
                                    problem_id=pid, temperature=0.8, top_p=0.95, max_tokens=1024, seed=None, n=32)
        pool = store.load(pool_id, pid, "olympiad-b-heldout", "qwen3.5-4b", "policy_primary")
        assert len(pool) == 32
        ordered = sorted(pool.samples, key=lambda s: s.sample_idx)
        feats = featurize(Probe(samples=ordered[:4]))
        action = predict_action(feats, threshold=0.6)
        stop_c = _majority_correct(ordered[:4], gold)
        sample_c = _majority_correct(ordered, gold)
        select_c = _any_correct(ordered, gold)
        is_correct = {"stop": stop_c, "sample": sample_c, "select": select_c, "abstain": False}[action]
        correct += int(is_correct)
        n += 1
        actions_used[action] = actions_used.get(action, 0) + 1

    acc = correct / n
    print(f"\n=== Fix #2 v2: hybrid fallback, threshold=0.6 (corrected selection criterion) ===")
    print(f"P4 accuracy: {acc:.4f} (n={n})  actions: {actions_used}")
    print(f"comparison: plain=0.3256 (-9.30pp vs best fixed 0.4186), Fix #1 (sample_x2)=0.3372 (-8.14pp), "
          f"Fix #2 v1 (threshold=0.4, flawed, no-op)=0.3256")
    print(f"Fix #2 v2 vs best fixed (0.4186): {100*(acc-0.4186):+.2f}pp")


if __name__ == "__main__":
    main()
