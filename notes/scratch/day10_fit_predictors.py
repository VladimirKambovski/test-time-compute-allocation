"""
Day 10: fit Detective (probe-feature logistic predictor) and Fortune
Teller (pre-hoc query-text embedding classifier) on the real, complete
P1 pool (500 MATH-500 problems, N=32) -- a genuine scaffold fit, not
Day 14's rigorous grouped-CV evaluation. Reports IN-SAMPLE accuracy
against the majority-class floor, purely to prove the pipeline produces
a real, sane number end to end.
"""

import json
import sys
import urllib.request
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, "src")

from marginal_token.controller.base import Probe  # noqa: E402
from marginal_token.controller.features import featurize  # noqa: E402
from marginal_token.controller.oracle_labels import oracle_action_label  # noqa: E402
from marginal_token.controller.predictor import DetectiveController, FortuneTellerController  # noqa: E402
from marginal_token.pools.store import PoolStore  # noqa: E402

POOL_ROOT = "results/pools"


def fetch_all_problem_data(unique_ids):
    """{unique_id: (problem_text, gold_answer)}"""
    wanted = set(unique_ids)
    found = {}
    offset = 0
    while offset < 500 and len(found) < len(wanted):
        url = (f"https://datasets-server.huggingface.co/rows?dataset=HuggingFaceH4/MATH-500"
               f"&config=default&split=test&offset={offset}&length=100")
        d = {}
        for _attempt in range(3):
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
                found[row["unique_id"]] = (row["problem"], row["answer"])
        offset += 100
    return found


def main():
    pool_meta = []
    for jsonl_path in sorted(Path(POOL_ROOT).glob("*/*.jsonl")):
        pool_id = jsonl_path.parent.name
        problem_id = jsonl_path.stem.replace("__", "/")
        pool_meta.append((pool_id, problem_id))
    print(f"{len(pool_meta)} pools found")

    problem_data = fetch_all_problem_data([pid for _, pid in pool_meta])
    print(f"fetched {len(problem_data)} problem texts + gold answers")

    store = PoolStore(POOL_ROOT)
    X_features, X_texts, y = [], [], []
    for pool_id, problem_id in pool_meta:
        pool = store.load(pool_id, problem_id, "math500", "qwen3.5-4b", "policy_primary")
        assert len(pool) == 32, f"{problem_id}: expected 32, got {len(pool)}"
        problem_text, gold = problem_data[problem_id]

        label = oracle_action_label(pool.samples, gold)
        probe_samples = sorted(pool.samples, key=lambda s: s.sample_idx)[:4]
        feats = featurize(Probe(samples=probe_samples))

        X_features.append(feats)
        X_texts.append(problem_text)
        y.append(label.action)

    print(f"\nlabel distribution: {dict(Counter(y))}")
    majority_class = Counter(y).most_common(1)[0][0]
    majority_floor = Counter(y)[majority_class] / len(y)
    print(f"majority-class floor ({majority_class}): {majority_floor:.4f}")

    print("\nfitting Detective (probe features -> action)...")
    detective = DetectiveController()
    detective.fit(X_features, y)

    import numpy as np
    from marginal_token.controller.features import FEATURE_NAMES
    arr = np.array([[f[name] for name in FEATURE_NAMES] for f in X_features], dtype=float)
    for i, name in enumerate(FEATURE_NAMES):
        col = arr[:, i]
        nan_mask = np.isnan(col)
        if nan_mask.any():
            fill = detective._impute_values[name]
            arr[nan_mask, i] = fill
    arr_scaled = detective.scaler.transform(arr)  # must scale -- the model was fit on scaled features
    detective_preds = detective.model.predict(arr_scaled)
    detective_acc = float(np.mean(detective_preds == np.array(y)))
    print(f"Detective in-sample accuracy: {detective_acc:.4f}")

    print("\nfitting Fortune Teller (query text embedding -> action)...")
    fortune_teller = FortuneTellerController()
    fortune_teller.fit(X_texts, y)
    embeddings = fortune_teller.embed(X_texts)
    ft_preds = fortune_teller.model.predict(embeddings)
    ft_acc = float(np.mean(ft_preds == np.array(y)))
    print(f"Fortune Teller in-sample accuracy: {ft_acc:.4f}")

    print(f"\n{'comparator':<28}{'in-sample accuracy':>20}")
    print(f"{'majority class':<28}{majority_floor:>20.4f}")
    print(f"{'Fortune Teller (pre-hoc)':<28}{ft_acc:>20.4f}")
    print(f"{'Detective (probe features)':<28}{detective_acc:>20.4f}")
    print("\n(All numbers are IN-SAMPLE, no held-out split -- this is Day 10's scaffold fit, "
          "not Day 14's rigorous grouped 5-fold CV evaluation. Expect these to drop once "
          "properly cross-validated.)")


if __name__ == "__main__":
    main()
