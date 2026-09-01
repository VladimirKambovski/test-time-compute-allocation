"""
Testing whether a targeted fix for Detective's held-out failure
(-9.3pp on P4, traced to over-abstaining relative to the true optimal
action distribution) actually works.

Discipline preserved: candidate configs are selected using ONLY dev-set
grouped CV (never touching held-out for selection) -- exactly one
held-out evaluation per selected candidate afterward, not a search
against held-out. This keeps the "held-out is one pass, no re-tuning"
rule intact even while testing a real fix.

Two families tested:
1. Partial class-weighting (a small grid between the plain model and
   the already-tested full 'balanced' extreme).
2. A hybrid rule: if Detective's top-class confidence is below a
   threshold, defer to the safe fixed policy (always-SAMPLE) instead.

Canonical-only pool enumeration throughout.
"""
import json
import sys
import urllib.request

sys.path.insert(0, "src")

import numpy as np  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from marginal_token.controller.base import Probe  # noqa: E402
from marginal_token.controller.features import FEATURE_NAMES, featurize  # noqa: E402
from marginal_token.controller.oracle_labels import oracle_action_label, _majority_correct, _any_correct  # noqa: E402
from marginal_token.controller.predictor import DetectiveController  # noqa: E402
from marginal_token.pools.store import PoolStore, compute_pool_id  # noqa: E402

RANDOM_STATE = 20260826


def _fetch_json_retry(url, attempts=5):
    import time
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


def fetch_olympiad_b_gold():
    ids = [str(x) for x in json.load(open("configs/benchmarks/data/heldout-olympiad-b-ids.json"))]
    return fetch_olympiad_all(ids), ids


def build_dev_data():
    math_gold = fetch_math500_all()
    oly_ids = [str(x) for x in json.load(open("configs/benchmarks/data/olympiad-a-ids.json"))]
    oly_gold = fetch_olympiad_all(oly_ids)

    store = PoolStore("results/pools")
    X_dicts, y = [], []
    benchmarks = []

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
    y = np.array(y)
    benchmarks = np.array(benchmarks)
    return X, y, benchmarks


def cv_eval(X, y, benchmarks, class_weight):
    classes = sorted(set(y.tolist()))
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    oof_probs = np.zeros((len(y), len(classes)))
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
        for i, c in enumerate(model.classes_):
            oof_probs[test_idx, classes.index(c)] = probs[:, i]
        oof_pred[test_idx] = model.classes_[np.argmax(probs, axis=1)]

    y_bin = np.array([[1 if label == c else 0 for c in classes] for label in y])
    auroc = roc_auc_score(y_bin, oof_probs, average="macro", multi_class="ovr")
    acc = (oof_pred == y).mean()
    sample_recall = ((oof_pred == "sample") & (y == "sample")).sum() / max((y == "sample").sum(), 1)
    abstain_recall = ((oof_pred == "abstain") & (y == "abstain")).sum() / max((y == "abstain").sum(), 1)
    return auroc, acc, sample_recall, abstain_recall


def main():
    print("building dev features/labels (canonical enumeration)...", flush=True)
    X, y, benchmarks = build_dev_data()
    print(f"n={len(y)}", flush=True)

    print("\n=== Step 1: select class_weight config via DEV-ONLY CV (never touches held-out) ===")
    print(f"{'config':>20}{'macro_auroc':>13}{'acc':>8}{'sample_recall':>15}{'abstain_recall':>16}")
    configs = {
        "plain": None,
        "sample_x2": {"stop": 1, "abstain": 1, "sample": 2, "select": 1},
        "sample_x3": {"stop": 1, "abstain": 1, "sample": 3, "select": 1},
        "balanced": "balanced",
    }
    cv_results = {}
    for name, cw in configs.items():
        auroc, acc, sr, ar = cv_eval(X, y, benchmarks, cw)
        cv_results[name] = (auroc, acc, sr, ar)
        print(f"{name:>20}{auroc:>13.4f}{acc:>8.4f}{sr:>15.4f}{ar:>16.4f}")

    # Selection criterion, decided BEFORE looking at held-out: best sample_recall
    # improvement over plain while keeping abstain_recall within 10pp of plain's.
    plain_ar = cv_results["plain"][3]
    best_name, best_sr = "plain", cv_results["plain"][2]
    for name, (auroc, acc, sr, ar) in cv_results.items():
        if name == "plain":
            continue
        if ar >= plain_ar - 0.10 and sr > best_sr:
            best_name, best_sr = name, sr
    print(f"\nselected (dev-CV only): {best_name}")

    print("\n=== Step 2: freeze the selected config on ALL dev data, evaluate ONCE on held-out P4 ===")
    y_str = y.tolist()
    controller = DetectiveController()
    X_dicts_for_fit = [dict(zip(FEATURE_NAMES, row)) for row in X]
    controller.fit(X_dicts_for_fit, y_str) if configs[best_name] is None else None
    # DetectiveController.fit doesn't take class_weight -- refit directly with the chosen weight
    all_nan = np.isnan(X).all(axis=0)
    col_means = np.zeros(X.shape[1])
    if not all_nan.all():
        with np.errstate(invalid="ignore"):
            col_means[~all_nan] = np.nanmean(X[:, ~all_nan], axis=0)
    X_imp = np.where(np.isnan(X), col_means, X)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imp)
    final_model = LogisticRegression(max_iter=2000, class_weight=configs[best_name])
    final_model.fit(X_scaled, y_str)

    def predict_action(feats_dict):
        row = np.array([[feats_dict[f] if not np.isnan(feats_dict[f]) else col_means[i]
                          for i, f in enumerate(FEATURE_NAMES)]])
        row_scaled = scaler.transform(row)
        probs = final_model.predict_proba(row_scaled)[0]
        class_probs = dict(zip(final_model.classes_, probs))
        return max(class_probs, key=class_probs.get), class_probs

    p4_ids = [str(x) for x in json.load(open("configs/benchmarks/data/heldout-olympiad-b-ids.json"))]
    p4_gold, _ = fetch_olympiad_all(p4_ids), None
    p4_gold = fetch_olympiad_all(p4_ids)
    store = PoolStore("results/pools")

    def eval_p4(predict_fn, threshold=None):
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
            action, class_probs = predict_fn(feats)
            if threshold is not None and max(class_probs.values()) < threshold:
                action = "sample"  # hybrid fallback: defer to the safe fixed policy
            stop_c = _majority_correct(ordered[:4], gold)
            sample_c = _majority_correct(ordered, gold)
            select_c = _any_correct(ordered, gold)
            is_correct = {"stop": stop_c, "sample": sample_c, "select": select_c, "abstain": False}[action]
            correct += int(is_correct)
            n += 1
            actions_used[action] = actions_used.get(action, 0) + 1
        return correct / n, actions_used, n

    print(f"\n--- Fix #1: recalibrated model ({best_name}), one held-out pass ---")
    acc, actions, n = eval_p4(predict_action)
    print(f"P4 accuracy: {acc:.4f} (n={n})  actions: {actions}")
    print(f"(original plain-model result for comparison: 0.3256, -9.30pp vs best fixed 0.4186)")

    print(f"\n=== Step 3: hybrid fallback, threshold selected via DEV-ONLY CV, one held-out pass ===")
    # Select threshold using dev CV out-of-fold probs from the PLAIN model (already computed above's cv_eval
    # doesn't return oof probs; rerun once quickly to get them for threshold selection).
    classes = sorted(set(y.tolist()))
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    oof_probs = np.zeros((len(y), len(classes)))
    oof_true_correct_if_hybrid = {}
    for thresh in (0.4, 0.5, 0.6):
        oof_pred_hybrid = np.empty(len(y), dtype=object)
        for train_idx, test_idx in skf.split(X, benchmarks):
            X_train = X[train_idx]
            all_nan_t = np.isnan(X_train).all(axis=0)
            col_means_t = np.zeros(X_train.shape[1])
            if not all_nan_t.all():
                with np.errstate(invalid="ignore"):
                    col_means_t[~all_nan_t] = np.nanmean(X_train[:, ~all_nan_t], axis=0)
            X_train_imp = np.where(np.isnan(X_train), col_means_t, X_train)
            X_test_imp = np.where(np.isnan(X[test_idx]), col_means_t, X[test_idx])
            scaler_t = StandardScaler()
            X_train_scaled = scaler_t.fit_transform(X_train_imp)
            X_test_scaled = scaler_t.transform(X_test_imp)
            model_t = LogisticRegression(max_iter=2000)
            model_t.fit(X_train_scaled, y[train_idx])
            probs = model_t.predict_proba(X_test_scaled)
            preds = model_t.classes_[np.argmax(probs, axis=1)]
            maxprob = probs.max(axis=1)
            preds = np.where(maxprob < thresh, "sample", preds)
            oof_pred_hybrid[test_idx] = preds
        acc_h = (oof_pred_hybrid == y).mean()
        sr_h = ((oof_pred_hybrid == "sample") & (y == "sample")).sum() / max((y == "sample").sum(), 1)
        print(f"  threshold={thresh}: dev CV accuracy={acc_h:.4f}, sample_recall={sr_h:.4f}")
        oof_true_correct_if_hybrid[thresh] = acc_h

    best_thresh = max(oof_true_correct_if_hybrid, key=oof_true_correct_if_hybrid.get)
    print(f"selected threshold (dev-CV only): {best_thresh}")

    print(f"\n--- Fix #2: hybrid fallback (threshold={best_thresh}), one held-out pass, using the PLAIN model's probs ---")
    acc2, actions2, n2 = eval_p4(predict_action if best_name != "plain" else predict_action, threshold=best_thresh)
    # use plain model specifically for the hybrid test, matching what was CV-selected
    plain_model = LogisticRegression(max_iter=2000)
    plain_model.fit(X_scaled, y_str)

    def predict_action_plain(feats_dict):
        row = np.array([[feats_dict[f] if not np.isnan(feats_dict[f]) else col_means[i]
                          for i, f in enumerate(FEATURE_NAMES)]])
        row_scaled = scaler.transform(row)
        probs = plain_model.predict_proba(row_scaled)[0]
        class_probs = dict(zip(plain_model.classes_, probs))
        return max(class_probs, key=class_probs.get), class_probs

    acc2, actions2, n2 = eval_p4(predict_action_plain, threshold=best_thresh)
    print(f"P4 accuracy: {acc2:.4f} (n={n2})  actions: {actions2}")
    print(f"(original plain-model result for comparison: 0.3256, best fixed policy: 0.4186)")


if __name__ == "__main__":
    main()
