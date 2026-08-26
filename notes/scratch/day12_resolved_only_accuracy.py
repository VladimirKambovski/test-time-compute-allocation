"""
Follow-up check: restrict Detective's predicted-action accuracy to only
the "resolved" problems (oracle label != abstain) -- does the predictor
actually choose well among STOP/SAMPLE/SELECT, separate from the easier
job of spotting the 36% of problems that are abstain-labeled? P1 only
(500 problems, no logprobs) -- memory-safe, matches the P1-only scripts
that ran fine earlier today.
"""

import json
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, "src")

import numpy as np

from marginal_token.controller.base import Probe  # noqa: E402
from marginal_token.controller.features import FEATURE_NAMES, featurize  # noqa: E402
from marginal_token.controller.oracle_labels import oracle_action_label  # noqa: E402
from marginal_token.controller.predictor import DetectiveController  # noqa: E402
from marginal_token.pools.store import PoolStore  # noqa: E402

POOL_ROOT = "results/pools"


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


def main():
    pool_meta = []
    for jsonl_path in sorted(Path(POOL_ROOT).glob("*/*.jsonl")):
        pool_id = jsonl_path.parent.name
        pid = jsonl_path.stem.replace("__", "/")
        if not pid.isdigit():  # P1 only
            pool_meta.append((pool_id, pid))
    assert len(pool_meta) == 500
    gold = fetch_gold([pid for _, pid in pool_meta])

    store = PoolStore(POOL_ROOT)
    X, y = [], []
    for pool_id, pid in pool_meta:
        pool = store.load(pool_id, pid, "math500", "qwen3.5-4b", "policy_primary")
        result = oracle_action_label(pool.samples, gold[pid])
        probe = sorted(pool.samples, key=lambda s: s.sample_idx)[:4]
        X.append(featurize(Probe(samples=probe)))
        y.append(result.action)
        del pool  # explicit, streaming-safe -- one pool in memory at a time

    detective = DetectiveController()
    detective.fit(X, y)

    arr = np.array([[f[n] for n in FEATURE_NAMES] for f in X], dtype=float)
    for i, name in enumerate(FEATURE_NAMES):
        mask = np.isnan(arr[:, i])
        if mask.any():
            arr[mask, i] = detective._impute_values[name]
    preds = detective.model.predict(detective.scaler.transform(arr))

    y_arr = np.array(y)
    overall_acc = float(np.mean(preds == y_arr))

    resolved_mask = y_arr != "abstain"
    resolved_acc = float(np.mean(preds[resolved_mask] == y_arr[resolved_mask]))
    n_resolved = int(resolved_mask.sum())

    # What does the model predict FOR the resolved-but-mispredicted cases?
    # (are mistakes mostly "predicted abstain when it wasn't," or genuine
    # stop/sample/select confusions?)
    from collections import Counter
    mistakes = Counter(zip(y_arr[resolved_mask][preds[resolved_mask] != y_arr[resolved_mask]],
                             preds[resolved_mask][preds[resolved_mask] != y_arr[resolved_mask]]))

    print(f"Overall accuracy (all 4 classes, in-sample): {overall_acc:.3f} (n=500)")
    print(f"Resolved-only accuracy (STOP/SAMPLE/SELECT, excluding ABSTAIN, in-sample): "
          f"{resolved_acc:.3f} (n={n_resolved})")
    print(f"\nlabel distribution among resolved: {dict(Counter(y_arr[resolved_mask]))}")
    print(f"mistakes on resolved cases (true -> predicted): {dict(mistakes)}")


if __name__ == "__main__":
    main()
