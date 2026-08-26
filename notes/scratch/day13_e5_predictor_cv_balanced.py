"""
Day 13 follow-up: same real grouped-CV Detective evaluation as
day13_e5_predictor_cv.py, but fits BOTH the plain multinomial logistic
regression (matching DetectiveController's frozen default exactly) AND
a class_weight='balanced' variant in the same CV loop, for a direct
side-by-side comparison -- does rebalancing recover SAMPLE/SELECT recall
without destroying the strong STOP/ABSTAIN/overall numbers?

Same data collection as before (streamed fresh -- pools haven't changed,
re-streaming is cheap and avoids relying on cross-process caching).
"""
import gc
import json
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, "src")

import numpy as np  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from marginal_token.controller.base import Probe  # noqa: E402
from marginal_token.controller.features import FEATURE_NAMES, featurize  # noqa: E402
from marginal_token.controller.oracle_labels import oracle_action_label  # noqa: E402
from marginal_token.pools.store import PoolStore  # noqa: E402

POOL_ROOT = "results/pools"


def fetch_math500_gold_all():
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
            found[row["unique_id"]] = row["answer"]
        offset += 100
    return found


def fetch_olympiad_gold_all():
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
                found[row_id] = row["final_answer"][0]
        offset += 100
    return found


def run_cv(X, y, benchmarks, classes, class_weight):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=20260826)
    n_classes = len(classes)
    oof_probs = np.zeros((len(y), n_classes))
    oof_pred = np.empty(len(y), dtype=object)

    for train_idx, test_idx in skf.split(X, benchmarks):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train = y[train_idx]

        all_nan_cols = np.isnan(X_train).all(axis=0)
        col_means = np.zeros(X_train.shape[1])
        if not all_nan_cols.all():
            with np.errstate(invalid="ignore"):
                col_means[~all_nan_cols] = np.nanmean(X_train[:, ~all_nan_cols], axis=0)
        X_train_imp = np.where(np.isnan(X_train), col_means, X_train)
        X_test_imp = np.where(np.isnan(X_test), col_means, X_test)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_imp)
        X_test_scaled = scaler.transform(X_test_imp)

        model = LogisticRegression(max_iter=2000, class_weight=class_weight)
        model.fit(X_train_scaled, y_train)

        probs = model.predict_proba(X_test_scaled)
        for i, cls in enumerate(model.classes_):
            oof_probs[test_idx, classes.index(cls)] = probs[:, i]
        oof_pred[test_idx] = model.classes_[np.argmax(probs, axis=1)]

    return oof_probs, oof_pred


def report(name, y, oof_probs, oof_pred, classes):
    print(f"\n=== {name} ===")
    y_bin = np.array([[1 if label == c else 0 for c in classes] for label in y])
    try:
        macro_auroc = roc_auc_score(y_bin, oof_probs, average="macro", multi_class="ovr")
        print(f"macro-AUROC: {macro_auroc:.4f}")
    except ValueError as e:
        print(f"macro-AUROC could not be computed: {e}")

    precision, recall, f1, support = precision_recall_fscore_support(y, oof_pred, labels=classes, zero_division=0)
    for c, p, r, f, s in zip(classes, precision, recall, f1, support):
        print(f"  {c}: precision={p:.3f} recall={r:.3f} f1={f:.3f} support={s}")

    overall_acc = (oof_pred == y).mean()
    print(f"overall accuracy: {overall_acc:.4f}")
    return overall_acc


def main():
    print("fetching gold answers...", flush=True)
    math_gold = fetch_math500_gold_all()
    oly_gold = fetch_olympiad_gold_all()
    print(f"got {len(math_gold)} MATH-500, {len(oly_gold)} OlympiadBench-A gold answers", flush=True)

    math_meta, oly_meta = [], []
    for jsonl_path in sorted(Path(POOL_ROOT).glob("*/*.jsonl")):
        pool_id = jsonl_path.parent.name
        problem_id = jsonl_path.stem.replace("__", "/")
        (oly_meta if problem_id.isdigit() else math_meta).append((pool_id, problem_id))

    store = PoolStore(POOL_ROOT)
    rows = []
    n_processed = 0

    for pool_id, pid in math_meta:
        if pid not in math_gold:
            continue
        pool = store.load(pool_id, pid, "math500", "qwen3.5-4b", "policy_primary")
        if len(pool) != 32:
            continue
        ordered = sorted(pool.samples, key=lambda s: s.sample_idx)
        probe = Probe(samples=ordered[:4])
        feats = featurize(probe)
        label = oracle_action_label(pool.samples, math_gold[pid]).action
        rows.append(("math500", label, feats))
        n_processed += 1
        if n_processed % 150 == 0:
            gc.collect()
            print(f"  {n_processed} processed", flush=True)

    for pool_id, pid in oly_meta:
        if pid not in oly_gold:
            continue
        pool = store.load(pool_id, pid, "olympiad-a", "qwen3.5-4b", "policy_primary")
        if len(pool) != 32:
            continue
        ordered = sorted(pool.samples, key=lambda s: s.sample_idx)
        probe = Probe(samples=ordered[:4])
        feats = featurize(probe)
        label = oracle_action_label(pool.samples, oly_gold[pid]).action
        rows.append(("olympiad-a", label, feats))
        n_processed += 1
        if n_processed % 150 == 0:
            gc.collect()
            print(f"  {n_processed} processed", flush=True)

    print(f"\ntotal usable problems: {n_processed}", flush=True)

    benchmarks = np.array([r[0] for r in rows])
    y = np.array([r[1] for r in rows])
    X = np.array([[r[2][f] for f in FEATURE_NAMES] for r in rows], dtype=float)
    classes = sorted(set(y.tolist()))

    probs_plain, pred_plain = run_cv(X, y, benchmarks, classes, class_weight=None)
    acc_plain = report("Detective (plain, frozen default -- matches predictor.py)", y, probs_plain, pred_plain, classes)

    probs_bal, pred_bal = run_cv(X, y, benchmarks, classes, class_weight="balanced")
    acc_bal = report("Detective (class_weight='balanced')", y, probs_bal, pred_bal, classes)

    print(f"\noverall accuracy: plain={acc_plain:.4f} vs balanced={acc_bal:.4f} "
          f"(balanced trades some overall accuracy for rare-class recall -- expected and often the right trade "
          f"when the rare classes are the ones that matter most, per HANDOFF's own framing)")


if __name__ == "__main__":
    main()
