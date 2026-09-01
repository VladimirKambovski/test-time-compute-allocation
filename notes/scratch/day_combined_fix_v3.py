"""
Third, final disclosed round of fix-testing (after Fix #1: partial
reweighting, Fix #2: confidence-threshold hybrid, both independently
landing on the same +1.16pp recovery). Question: does pushing further
in the same direction, or combining both levers, do better -- or does
it plateau/reverse? Selected via dev-only CV, ONE held-out touch for
the winning combined candidate. This is the last round of this
exploration, not an open-ended search.
"""
import json
import sys
import time
import urllib.request

sys.path.insert(0, "src")

import numpy as np  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from marginal_token.controller.base import Probe  # noqa: E402
from marginal_token.controller.features import FEATURE_NAMES, featurize  # noqa: E402
from marginal_token.controller.oracle_labels import oracle_action_label, _majority_correct, _any_correct  # noqa: E402
from marginal_token.pools.store import PoolStore, compute_pool_id  # noqa: E402

RANDOM_STATE = 20260826


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
    X_dicts, y, benchmarks = [], [], []
    for pid, gold in math_gold.items():
        pool_id = compute_pool_id(policy_ref="qwen3.5-4b", backend_ref="policy_primary", benchmark_id="math500",
                                    problem_id=pid, temperature=0.8, top_p=0.95, max_tokens=1024, seed=None, n=32)
        pool = store.load(pool_id, pid, "math500", "qwen3.5-4b", "policy_primary")
        assert len(pool) == 32
        ordered = sorted(pool.samples, key=lambda s: s.sample_idx)
        X_dicts.append(featurize(Probe(samples=ordered[:4])))
        y.append(oracle_action_label(pool.samples, gold).action)
        benchmarks.append("math500")
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
        benchmarks.append("olympiad-a")
    X = np.array([[d[f] for f in FEATURE_NAMES] for d in X_dicts], dtype=float)
    return X, np.array(y), np.array(benchmarks)


def cv_eval_combined(X, y, benchmarks, class_weight, threshold):
    classes = sorted(set(y.tolist()))
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    oof_pred = np.empty(len(y), dtype=object)
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
        model = LogisticRegression(max_iter=2000, class_weight=class_weight)
        model.fit(X_train_scaled, y[train_idx])
        probs = model.predict_proba(X_test_scaled)
        preds = model.classes_[np.argmax(probs, axis=1)]
        if threshold is not None:
            preds = np.where(probs.max(axis=1) < threshold, "sample", preds)
        oof_pred[test_idx] = preds
    acc = (oof_pred == y).mean()
    sr = ((oof_pred == "sample") & (y == "sample")).sum() / max((y == "sample").sum(), 1)
    ar = ((oof_pred == "abstain") & (y == "abstain")).sum() / max((y == "abstain").sum(), 1)
    return acc, sr, ar


def main():
    print("rebuilding dev features/labels...", flush=True)
    X, y, benchmarks = build_dev_data()
    print(f"n={len(y)}", flush=True)

    candidates = [
        ("sample_x2_only", {"stop": 1, "abstain": 1, "sample": 2, "select": 1}, None),
        ("sample_x4_only", {"stop": 1, "abstain": 1, "sample": 4, "select": 1}, None),
        ("thresh_0.7_only", None, 0.7),
        ("sample_x2+thresh_0.5", {"stop": 1, "abstain": 1, "sample": 2, "select": 1}, 0.5),
        ("sample_x2+thresh_0.6", {"stop": 1, "abstain": 1, "sample": 2, "select": 1}, 0.6),
    ]
    print(f"\n{'config':>24}{'acc':>8}{'sample_recall':>15}{'abstain_recall':>16}")
    results = {}
    for name, cw, th in candidates:
        acc, sr, ar = cv_eval_combined(X, y, benchmarks, cw, th)
        results[name] = (acc, sr, ar, cw, th)
        print(f"{name:>24}{acc:>8.4f}{sr:>15.4f}{ar:>16.4f}")

    plain_acc = 0.9244  # from the earlier run, for reference
    # Selection: best sample_recall among configs with accuracy >= plain_acc - 0.03
    best_name = max(
        (n for n in results if results[n][0] >= plain_acc - 0.03),
        key=lambda n: results[n][1],
        default=None,
    )
    print(f"\nselected (dev-CV only, accuracy >= {plain_acc-0.03:.4f}): {best_name}")

    acc, sr, ar, cw, th = results[best_name]

    all_nan = np.isnan(X).all(axis=0)
    col_means = np.zeros(X.shape[1])
    if not all_nan.all():
        with np.errstate(invalid="ignore"):
            col_means[~all_nan] = np.nanmean(X[:, ~all_nan], axis=0)
    X_imp = np.where(np.isnan(X), col_means, X)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imp)
    final_model = LogisticRegression(max_iter=2000, class_weight=cw)
    final_model.fit(X_scaled, y.tolist())

    def predict_action(feats_dict):
        row = np.array([[feats_dict[f] if not np.isnan(feats_dict[f]) else col_means[i]
                          for i, f in enumerate(FEATURE_NAMES)]])
        row_scaled = scaler.transform(row)
        probs = final_model.predict_proba(row_scaled)[0]
        class_probs = dict(zip(final_model.classes_, probs))
        action = max(class_probs, key=class_probs.get)
        if th is not None and max(class_probs.values()) < th:
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
        action = predict_action(feats)
        stop_c = _majority_correct(ordered[:4], gold)
        sample_c = _majority_correct(ordered, gold)
        select_c = _any_correct(ordered, gold)
        is_correct = {"stop": stop_c, "sample": sample_c, "select": select_c, "abstain": False}[action]
        correct += int(is_correct)
        n += 1
        actions_used[action] = actions_used.get(action, 0) + 1

    print(f"\n=== Fix #3 (combined/pushed-further): {best_name}, one held-out pass ===")
    print(f"P4 accuracy: {correct/n:.4f} (n={n})  actions: {actions_used}")
    print(f"vs best fixed (0.4186): {100*(correct/n-0.4186):+.2f}pp")
    print(f"(prior fixes both landed at 0.3372, -8.14pp)")


if __name__ == "__main__":
    main()
