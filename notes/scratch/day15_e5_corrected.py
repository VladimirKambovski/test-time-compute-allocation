"""
CORRECTED rerun of day13_e5_predictor_cv.py / day14_e5_full_comparators.py,
prompted by discovering a real bug: those scripts (and day15's original
E7/E8 draft) enumerated problems via a plain directory glob over
results/pools/*/*.jsonl, with NO canonical-pool_id verification. 49
problem_ids have multiple valid (len==32) on-disk pool directories
(confirmed via /tmp/dupe_check.py) -- stray/historical pools that
happen to also have exactly 32 samples, NOT the real P1/P2 data. The
naive glob approach double/triple-counted those 49 problems.

Fix: enumerate the CANONICAL 500 MATH-500 + 300 OlympiadBench-A ids
directly (from the dataset + configs/benchmarks/data/olympiad-a-ids.json),
compute each one's canonical pool_id via compute_pool_id with the known
P1/P2 decode config (temp=0.8, top_p=0.95, max_tokens=1024, seed=None,
n=32) -- verified 2026-08-2X: ALL 800 canonical pools exist on disk, 0
missing. Never trust the glob for problem identity again.

Same CV methodology (grouped-by-problem, stratified-by-benchmark,
random_state=20260826) as the original run, so this is a clean,
comparable correction, not a different experiment.
"""
import gc
import json
import sys
import time
import urllib.request

sys.path.insert(0, "src")

import numpy as np  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from marginal_token.controller.base import Probe  # noqa: E402
from marginal_token.controller.features import FEATURE_NAMES, featurize  # noqa: E402
from marginal_token.controller.oracle_labels import oracle_action_label  # noqa: E402
from marginal_token.pools.store import PoolStore, compute_pool_id  # noqa: E402

POOL_ROOT = "results/pools"
RANDOM_STATE = 20260826


def fetch_math500_all():
    found = {}
    offset = 0
    while offset < 500:
        url = (f"https://datasets-server.huggingface.co/rows?dataset=HuggingFaceH4/MATH-500"
               f"&config=default&split=test&offset={offset}&length=100")
        d = json.load(urllib.request.urlopen(url, timeout=30))
        for r in d["rows"]:
            row = r["row"]
            found[row["unique_id"]] = (row["problem"], row["answer"])
        offset += 100
    return found


def fetch_olympiad_all(wanted_ids):
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
                found[row_id] = (row["question"], row["final_answer"][0])
        offset += 100
    return found


def main():
    print("fetching MATH-500 (all 500) + OlympiadBench-A (canonical 300 ids)...", flush=True)
    math_data = fetch_math500_all()
    oly_ids = [str(x) for x in json.load(open("configs/benchmarks/data/olympiad-a-ids.json"))]
    oly_data = fetch_olympiad_all(oly_ids)
    print(f"got {len(math_data)} MATH-500, {len(oly_data)}/{len(oly_ids)} OlympiadBench-A usable", flush=True)

    store = PoolStore(POOL_ROOT)
    rows = []
    n_processed = 0

    for pid, (text, gold) in math_data.items():
        pool_id = compute_pool_id(policy_ref="qwen3.5-4b", backend_ref="policy_primary", benchmark_id="math500",
                                    problem_id=pid, temperature=0.8, top_p=0.95, max_tokens=1024, seed=None, n=32)
        pool = store.load(pool_id, pid, "math500", "qwen3.5-4b", "policy_primary")
        assert len(pool) == 32, f"{pid}: canonical pool has {len(pool)} samples, expected 32"
        ordered = sorted(pool.samples, key=lambda s: s.sample_idx)
        probe = Probe(samples=ordered[:4])
        feats = featurize(probe)
        label = oracle_action_label(pool.samples, gold).action
        rows.append(("math500", label, feats, text))
        n_processed += 1
        if n_processed % 150 == 0:
            gc.collect()
            print(f"  {n_processed} processed", flush=True)

    for pid in oly_ids:
        if pid not in oly_data:
            continue
        text, gold = oly_data[pid]
        pool_id = compute_pool_id(policy_ref="qwen3.5-4b", backend_ref="policy_primary", benchmark_id="olympiad-a",
                                    problem_id=pid, temperature=0.8, top_p=0.95, max_tokens=1024, seed=None, n=32)
        pool = store.load(pool_id, pid, "olympiad-a", "qwen3.5-4b", "policy_primary")
        assert len(pool) == 32, f"{pid}: canonical pool has {len(pool)} samples, expected 32"
        ordered = sorted(pool.samples, key=lambda s: s.sample_idx)
        probe = Probe(samples=ordered[:4])
        feats = featurize(probe)
        label = oracle_action_label(pool.samples, gold).action
        rows.append(("olympiad-a", label, feats, text))
        n_processed += 1
        if n_processed % 150 == 0:
            gc.collect()
            print(f"  {n_processed} processed", flush=True)

    print(f"\ntotal usable problems (canonical, deduped): {n_processed}", flush=True)
    from collections import Counter
    print(f"class distribution: {dict(Counter(r[1] for r in rows))}")

    benchmarks = np.array([r[0] for r in rows])
    y = np.array([r[1] for r in rows])
    X = np.array([[r[2][f] for f in FEATURE_NAMES] for r in rows], dtype=float)
    texts = [r[3] for r in rows]
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
        model = LogisticRegression(max_iter=2000)
        model.fit(X_train_scaled, y[train_idx])
        probs = model.predict_proba(X_test_scaled)
        for i, c in enumerate(model.classes_):
            oof_probs[test_idx, classes.index(c)] = probs[:, i]
        oof_pred[test_idx] = model.classes_[np.argmax(probs, axis=1)]

    y_bin = np.array([[1 if label == c else 0 for c in classes] for label in y])
    detective_auroc = roc_auc_score(y_bin, oof_probs, average="macro", multi_class="ovr")
    print(f"\nDetective macro-AUROC (CORRECTED, canonical-only): {detective_auroc:.4f}  "
          f"(previous buggy run: 0.8797)")

    precision, recall, f1, support = precision_recall_fscore_support(y, oof_pred, labels=classes, zero_division=0)
    for c, p, r, f, s in zip(classes, precision, recall, f1, support):
        print(f"  {c}: precision={p:.3f} recall={r:.3f} f1={f:.3f} support={s}")
    overall_acc = (oof_pred == y).mean()
    majority_class = Counter(y.tolist()).most_common(1)[0][0]
    majority_acc = (y == majority_class).mean()
    print(f"overall accuracy: {overall_acc:.4f} (previous buggy run: 0.9009)")
    print(f"trivial majority-class accuracy: {majority_acc:.4f}")

    print("\nfitting Fortune Teller (pre-hoc)...", flush=True)
    from sentence_transformers import SentenceTransformer
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = np.asarray(embedder.encode(texts, show_progress_bar=False))
    ft_oof_probs = np.zeros((len(y), len(classes)))
    for train_idx, test_idx in skf.split(X, benchmarks):
        model = LogisticRegression(max_iter=2000)
        model.fit(embeddings[train_idx], y[train_idx])
        probs = model.predict_proba(embeddings[test_idx])
        for i, c in enumerate(model.classes_):
            ft_oof_probs[test_idx, classes.index(c)] = probs[:, i]
    ft_auroc = roc_auc_score(y_bin, ft_oof_probs, average="macro", multi_class="ovr")
    print(f"Fortune Teller macro-AUROC (CORRECTED): {ft_auroc:.4f}  (previous buggy run: 0.7263)")

    diff = detective_auroc - ft_auroc
    print(f"\nH2 (Detective >= 0.70): {detective_auroc:.4f} -> {'ACCEPT' if detective_auroc>=0.70 else ('WEAK' if detective_auroc>=0.60 else 'REJECT')}")
    print(f"H3 (Detective - Fortune Teller >= 0.05): {diff:.4f} -> {'ACCEPT' if diff>=0.05 else 'REJECT'}")


if __name__ == "__main__":
    main()
