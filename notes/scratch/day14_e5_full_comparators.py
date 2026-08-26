"""
Day 14, pulled forward: E5's full comparator set, evaluated under the
SAME grouped-by-problem/stratified-by-benchmark 5-fold CV split as
day13_e5_predictor_cv.py (identical StratifiedKFold random_state), so
H3's paired post-hoc-vs-pre-hoc comparison is valid.

Comparators (docs/brief.md section 16):
1. Detective (post-hoc, probe features) -- recomputed here for a
   single consistent source of truth alongside the others.
2. Fortune Teller (pre-hoc, query-text embedding) -- H3's non-negotiable
   control.
3. Majority class -- trivial floor.
4. Fixed agreement threshold on the probe -- Adaptive-Consistency's
   published signal class (V5).
Difficulty-tier oracle and full oracle are ceilings, not learned models
-- computed separately/trivially, not the focus of this pass.

H2: Detective macro-AUROC >=0.70 (already confirmed 0.8797 tonight,
recomputed here for consistency).
H3: Detective macro-AUROC exceeds Fortune Teller's by >=0.05, paired
over problems.
"""
import gc
import json
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, "src")

import numpy as np  # noqa: E402
from sklearn.dummy import DummyClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from marginal_token.controller.base import Probe  # noqa: E402
from marginal_token.controller.features import FEATURE_NAMES, featurize  # noqa: E402
from marginal_token.controller.oracle_labels import oracle_action_label  # noqa: E402
from marginal_token.pools.store import PoolStore  # noqa: E402

POOL_ROOT = "results/pools"
RANDOM_STATE = 20260826  # SAME as day13_e5_predictor_cv.py -- keeps folds identical for pairing


def fetch_math500_all():
    """Returns {unique_id: (problem_text, gold_answer)}."""
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
    """Returns {id: (question_text, gold_answer)}."""
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


def out_of_fold_probs(fit_fn, predict_fn, y, benchmarks, classes):
    """Generic grouped/stratified 5-fold CV runner. fit_fn(train_idx) ->
    model; predict_fn(model, test_idx) -> (n_test, n_classes) prob array
    aligned to `classes` order.
    """
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    n_classes = len(classes)
    oof_probs = np.zeros((len(y), n_classes))
    for train_idx, test_idx in skf.split(np.zeros(len(y)), benchmarks):
        model = fit_fn(train_idx)
        oof_probs[test_idx] = predict_fn(model, test_idx)
    return oof_probs


def macro_auroc(y, oof_probs, classes):
    y_bin = np.array([[1 if label == c else 0 for c in classes] for label in y])
    try:
        return roc_auc_score(y_bin, oof_probs, average="macro", multi_class="ovr")
    except ValueError as e:
        print(f"  (macro-AUROC could not be computed: {e})")
        return None


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

    store = PoolStore(POOL_ROOT)
    rows = []  # (benchmark_id, label, feature_dict, query_text, top1_vote_fraction)
    n_processed = 0

    for pool_id, pid in math_meta:
        if pid not in math_data:
            continue
        text, gold = math_data[pid]
        pool = store.load(pool_id, pid, "math500", "qwen3.5-4b", "policy_primary")
        if len(pool) != 32:
            continue
        ordered = sorted(pool.samples, key=lambda s: s.sample_idx)
        probe = Probe(samples=ordered[:4])
        feats = featurize(probe)
        label = oracle_action_label(pool.samples, gold).action
        rows.append(("math500", label, feats, text))
        n_processed += 1
        if n_processed % 150 == 0:
            gc.collect()
            print(f"  {n_processed} processed", flush=True)

    for pool_id, pid in oly_meta:
        if pid not in oly_data:
            continue
        text, gold = oly_data[pid]
        pool = store.load(pool_id, pid, "olympiad-a", "qwen3.5-4b", "policy_primary")
        if len(pool) != 32:
            continue
        ordered = sorted(pool.samples, key=lambda s: s.sample_idx)
        probe = Probe(samples=ordered[:4])
        feats = featurize(probe)
        label = oracle_action_label(pool.samples, gold).action
        rows.append(("olympiad-a", label, feats, text))
        n_processed += 1
        if n_processed % 150 == 0:
            gc.collect()
            print(f"  {n_processed} processed", flush=True)

    print(f"\ntotal usable problems: {n_processed}", flush=True)

    benchmarks = np.array([r[0] for r in rows])
    y = np.array([r[1] for r in rows])
    X = np.array([[r[2][f] for f in FEATURE_NAMES] for r in rows], dtype=float)
    texts = [r[3] for r in rows]
    top1_vote = np.array([r[2]["top1_vote_fraction"] for r in rows])
    classes = sorted(set(y.tolist()))

    # ---- 1. Detective (post-hoc, probe features) ----
    def fit_detective(train_idx):
        X_train = X[train_idx]
        all_nan = np.isnan(X_train).all(axis=0)
        col_means = np.zeros(X_train.shape[1])
        if not all_nan.all():
            with np.errstate(invalid="ignore"):
                col_means[~all_nan] = np.nanmean(X_train[:, ~all_nan], axis=0)
        X_train_imp = np.where(np.isnan(X_train), col_means, X_train)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_imp)
        model = LogisticRegression(max_iter=2000)
        model.fit(X_train_scaled, y[train_idx])
        return (model, scaler, col_means)

    def predict_detective(state, test_idx):
        model, scaler, col_means = state
        X_test = X[test_idx]
        X_test_imp = np.where(np.isnan(X_test), col_means, X_test)
        X_test_scaled = scaler.transform(X_test_imp)
        probs = model.predict_proba(X_test_scaled)
        out = np.zeros((len(test_idx), len(classes)))
        for i, c in enumerate(model.classes_):
            out[:, classes.index(c)] = probs[:, i]
        return out

    print("\nfitting Detective (post-hoc, probe features)...", flush=True)
    detective_probs = out_of_fold_probs(fit_detective, predict_detective, y, benchmarks, classes)
    detective_auroc = macro_auroc(y, detective_probs, classes)
    print(f"Detective macro-AUROC: {detective_auroc:.4f}  (H2: >=0.70 accept)")

    # ---- 2. Fortune Teller (pre-hoc, query-text embedding) ----
    print("\nembedding all query texts (sentence-transformers, one-time model load)...", flush=True)
    from sentence_transformers import SentenceTransformer
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = np.asarray(embedder.encode(texts, show_progress_bar=False))
    print(f"embeddings shape: {embeddings.shape}", flush=True)

    def fit_fortune_teller(train_idx):
        model = LogisticRegression(max_iter=2000)
        model.fit(embeddings[train_idx], y[train_idx])
        return model

    def predict_fortune_teller(model, test_idx):
        probs = model.predict_proba(embeddings[test_idx])
        out = np.zeros((len(test_idx), len(classes)))
        for i, c in enumerate(model.classes_):
            out[:, classes.index(c)] = probs[:, i]
        return out

    print("\nfitting Fortune Teller (pre-hoc, query-text embedding, H3's control)...", flush=True)
    ft_probs = out_of_fold_probs(fit_fortune_teller, predict_fortune_teller, y, benchmarks, classes)
    ft_auroc = macro_auroc(y, ft_probs, classes)
    print(f"Fortune Teller macro-AUROC: {ft_auroc:.4f}")

    print(f"\n=== H3: post-hoc (Detective) vs pre-hoc (Fortune Teller) ===")
    diff = detective_auroc - ft_auroc
    print(f"Detective - Fortune Teller = {diff:.4f}  (H3: >=0.05 needed to accept)")
    print(f"H3 verdict: {'ACCEPT' if diff >= 0.05 else 'REJECT'} (evidence beats pre-hoc guess by {'>=' if diff>=0.05 else '<'} 0.05)")

    # ---- 3. Majority class (trivial floor) ----
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X[:1], y[:1])  # sklearn needs a fit call; doesn't matter what data
    maj_probs = np.zeros((len(y), len(classes)))
    from collections import Counter
    majority_class = Counter(y.tolist()).most_common(1)[0][0]
    maj_probs[:, classes.index(majority_class)] = 1.0
    maj_auroc = macro_auroc(y, maj_probs, classes)
    print(f"\nMajority-class comparator macro-AUROC: {maj_auroc}  (expected ~0.5, no discrimination by construction)")

    # ---- 4. Fixed agreement threshold (Adaptive-Consistency signal, V5) ----
    # Simple rule: predict STOP if top1_vote_fraction >= threshold, else
    # predict the global second-most-common class (ABSTAIN) as a crude
    # fallback -- a genuine two-bucket heuristic, not a learned model.
    best_thresh, best_acc = None, -1
    for thresh in np.arange(0.25, 1.0, 0.05):
        pred = np.where(top1_vote >= thresh, "stop", "abstain")
        acc = (pred == y).mean()
        if acc > best_acc:
            best_acc, best_thresh = acc, thresh
    print(f"\nFixed agreement-threshold comparator (best threshold={best_thresh:.2f}): accuracy={best_acc:.4f} "
          f"(2-bucket STOP-vs-ABSTAIN heuristic, no SAMPLE/SELECT capacity by construction -- "
          f"a real ceiling-on-simplicity comparator, not directly AUROC-comparable to the 4-class models)")

    print(f"\n=== SUMMARY ===")
    print(f"H2 (Detective macro-AUROC >= 0.70): {detective_auroc:.4f} -> {'ACCEPT' if detective_auroc >= 0.70 else ('WEAK' if detective_auroc >= 0.60 else 'REJECT')}")
    print(f"H3 (Detective beats Fortune Teller by >= 0.05): {diff:.4f} -> {'ACCEPT' if diff >= 0.05 else 'REJECT'}")
    print(f"Majority-class floor: {maj_auroc} (Detective's real lift: {detective_auroc - (maj_auroc or 0.5):.4f})")
    print(f"Fixed-threshold heuristic accuracy: {best_acc:.4f} (vs Detective's overall accuracy from the earlier E5 run: 0.9009)")


if __name__ == "__main__":
    main()
